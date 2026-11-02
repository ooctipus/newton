# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The one writer of the general placement (FL3.3): every row a seed and a law, whitened by the walk.

A dense row of the general placement (case K6) holds its unit source (:func:`.dense_rows.unit_source`), which its
producer records with the row's slot and records, in place of a Jacobian. After the row pass, the writer takes every
row of a world on the world's lanes (:func:`walked_rows_kernel`):

- **the seed**, from the row's law (:data:`.dense_rows.UNIT_CONTACT` and the other kinds): a contact row's wrench on
  each of its bodies, ``(d, (p - o) x d)`` at the body's frame ``o`` at the step's poses (its spin row's ``(0, n)``),
  the second body's negated; a link row's unit wrench at its two pieces' centres of mass, the shadow's negated; a joint
  limit's and a velocity limit's covector ``+-e_c``, a mimic's ``e_f - c e_l``;
- **the walk** (:func:`.tree_factor.walk_paths_prelude`): each seed walked up its root path into the packed sink at its
  side's entries, stored, a second seed on one tree adding where their paths meet. A wrench enters as ``J`` at the
  step's frames (:meth:`.tree_factor.TreeFactor.walk_step`) and the refresh's factor whitens it, so a step between
  refreshes whitens the step's ``J`` by the held factor, as the dense rows do. A free rigid body is a one-body tree:
  its side is a six-step walk;
- **the header**: the diagonal ``|z|^2 + cfm`` (the walks' sums of squares, or the row's entries where two seeds share
  a side or a side slips), the incident velocity ``J v_hat`` (the step's ``J`` entries each walk passes, against
  ``v_hat``; a free rigid body's six) and the rhs by the row law (:func:`.kernels.row_bias`).

The free bodies' rows (the bodies sink, ``mf_Z_a`` and ``mf_Z_b``) take the same seeds and walks
(:func:`write_free_body_rows`): each contact row's records by the contact law, its two sides' six-step walks and its
incident velocity; the rows' build (:func:`.response_builds_rows.free_body_rows_kernel`) then stores the diagonal and
the rhs.

No ``J`` is formed. The rows and the sweeps' leave (:class:`.sweep_routes.FactorCoordinates`, ``walk``; the paired
factor sweep's) share the walk's one frame (FL3_design §1.2).
"""

from __future__ import annotations

from functools import cache

import warp as wp

from .contact_input import ContactInput, contact_frame
from .contact_laws import CONTACT_SPIN_ROW, contact_row_frame, contact_seed, spin_seed
from .dense_rows import (
    UNIT_CONTACT,
    UNIT_LIMIT,
    UNIT_LINK,
    UNIT_MIMIC,
    UNIT_SHIFT,
    UNIT_VELOCITY,
    DenseRows,
    MfRows,
    RowSupport,
    jacobian_row,
    unit_source,
)
from .device_preludes import dual_native, lane_group
from .friction_patches import PatchRowLaw
from .kernel_template import python_kernel
from .kernels import PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT, free_body_contact, free_body_contact_records, row_bias
from .response_builds_rows import close_block, close_row, free_body_row_header
from .row_build import write_contact_row
from .row_laws import LINK_ROWS
from .solver_fields import Preelim
from .tree_factor import WalkSlips, WalkTrees, release_path, walk_into, walk_release_tree

__all__ = [
    "free_body_headers_kernel",
    "walked_rows_kernel",
    "write_contact_units",
    "write_free_body_rows",
    "writer_kernel",
    "writer_lanes",
]


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
    add: int,
    packed_Z: wp.array3d[float],
    slot: int,
) -> wp.vec2:
    """Walk a row's seed at ``body``, the wrench of ``sign`` along ``direction`` at ``point`` (a pure ``torque`` about
    it), into its side's entries of row ``slot``: stored, or added with ``add`` (a second seed on the same side). Return
    its part of the incident velocity ``J v_hat`` (the step's ``J`` entries the walk passes) and the sum of its
    footprints' squares. A body without response coordinates adds nothing."""
    if body < 0 or art < 0:
        return wp.vec2(0.0)
    if support.dofs[art] <= 0:
        return wp.vec2(0.0)
    at = jacobian_row(support, 0, art, sides)
    wrench = body_wrench(walk, body, sign * direction, point, torque)
    return walk_body(walk, art, body, wrench, v_hat, add, packed_Z, at[0], slot, at[1])


@wp.func
def body_wrench(walk: WalkTrees, body: int, direction: wp.vec3, point: wp.vec3, torque: bool) -> wp.spatial_vector:
    """The seed of a row acting on ``body`` along ``direction`` at ``point`` (a pure ``torque`` about it), by the
    contact law (:func:`.contact_laws.contact_seed`, :func:`.contact_laws.spin_seed`): a wrench at the body's frame at
    the step's poses (:attr:`.tree_factor.WalkTrees.now_link`), where the walk's columns ``now`` hold the step's
    ``J``. The lever arm is the world points' difference here; the law takes it from wherever it is formed."""
    if torque:
        return spin_seed(direction)
    j = walk.body_joint[body]
    frame = wp.vec3(walk.now_link[j, 4], walk.now_link[j, 5], walk.now_link[j, 6])
    return contact_seed(direction, point - frame)


@wp.func
def walk_body(
    walk: WalkTrees,
    art: int,
    body: int,
    wrench: wp.spatial_vector,
    v_hat: wp.array[float],
    add: int,
    sink: wp.array3d[float],
    line: int,
    slot: int,
    entry: int,
) -> wp.vec2:
    """Walk ``wrench`` at ``body``'s frame up articulation ``art``'s tree into ``sink[line, slot, entry + c]``
    (:func:`.tree_factor.walk_into`): the packed rows' sides and the free bodies' six entries take their footprints
    from here."""
    j0 = walk.joint_start[art]
    return walk_into(
        walk,
        j0,
        walk.dof_start[art],
        walk.joint_end[art] - j0,
        walk.dofs[art],
        walk.body_joint[body] - j0,
        wrench,
        -1,
        0.0,
        v_hat,
        add,
        sink,
        line,
        slot,
        entry,
    )


@wp.func
def walk_covector(
    walk: WalkTrees,
    support: RowSupport,
    sides: wp.vec2i,
    art: int,
    dof: int,
    value: float,
    v_hat: wp.array[float],
    add: int,
    packed_Z: wp.array3d[float],
    slot: int,
) -> wp.vec2:
    """Walk a row's covector ``value e_dof`` (a joint row's seed) into its side's entries of row ``slot`` (added with
    ``add``); return its part of the incident velocity, ``value v_hat[dof]``, and the sum of its footprints' squares."""
    if art < 0:
        return wp.vec2(0.0)
    if support.dofs[art] <= 0:
        return wp.vec2(0.0)
    d0 = walk.dof_start[art]
    j0 = walk.joint_start[art]
    at = jacobian_row(support, 0, art, sides)
    return walk_into(
        walk,
        j0,
        d0,
        walk.joint_end[art] - j0,
        walk.dofs[art],
        walk.dof_body[dof],
        wp.spatial_vector(),
        dof - d0,
        value,
        v_hat,
        add,
        packed_Z,
        at[0],
        slot,
        at[1],
    )


@wp.func
def slips_on(walk: WalkTrees, art: int) -> int:
    """Whether articulation ``art`` (-1 for none) slips this step: its rows take the release."""
    if art < 0:
        return 0
    return walk.release.count[art]


@wp.func
def release_side(
    walk: WalkTrees, support: RowSupport, sides: wp.vec2i, art: int, packed_Z: wp.array3d[float], slot: int
):
    """The slips' release on articulation ``art``'s side of row ``slot``: ``z <- R z``, the root-most slip first, each
    along its path leaf first (:func:`.tree_factor.walk_release_tree`), the decided operator's row on the all-``K``
    walk. The packed side holds the whole tree, and ``R`` keeps every row's support."""
    if art < 0:
        return
    at = jacobian_row(support, 0, art, sides)
    for j in range(walk.release.count[art]):
        release_path(walk.release, art, j, packed_Z, at[0], slot, at[1])


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
def write_free_body_rows(
    mf: MfRows,
    data: ContactInput,
    total_threads: int,
    has_target_velocity: int,
    law: PatchRowLaw,
    walk: WalkTrees,
    v_hat: wp.array[float],
):
    """The bodies sink: each free-body contact row (``path`` 1, grid-stride over the contacts) by its law, its records
    (:func:`.kernels.free_body_contact_records`), each responding body's seed walked into its six entries (``mf_Z_a``,
    ``mf_Z_b``; the second body's negated, a spin row a torque about the normal) and its incident velocity ``J v_hat``.
    The rows' build stores their diagonals and rhs (:func:`.response_builds_rows.free_body_rows_kernel`)."""
    for c in range(wp.tid(), data.count[0], total_threads):
        slot = data.slot[c]
        if data.path[c] != 1 or slot < 0:
            continue
        mf_line = mf.first[data.world[c]]
        art_a = data.art0[c]
        art_b = data.art1[c]
        body_a, body_b, normal, point, phi, tangent0, tangent1, friction_point = contact_frame(data, law, c)
        response_a, response_b, mu, restitution, stiffness, rows = free_body_contact(data, c, body_a, body_b)
        for row in range(rows):
            direction, row_point = contact_row_frame(row, normal, tangent0, tangent1, point, friction_point)
            spin = row == CONTACT_SPIN_ROW
            incident = float(0.0)
            if response_a >= 0:
                wrench = body_wrench(walk, response_a, direction, row_point, spin)
                incident += walk_body(walk, art_a, response_a, wrench, v_hat, 0, mf.Z_a, mf_line, slot + row, 0)[0]
            if response_b >= 0:
                wrench = body_wrench(walk, response_b, -direction, row_point, spin)
                incident += walk_body(walk, art_b, response_b, wrench, v_hat, 0, mf.Z_b, mf_line, slot + row, 0)[0]
            mf.incident[mf_line, slot + row] = incident
            free_body_contact_records(
                mf,
                data,
                law,
                has_target_velocity,
                c,
                slot,
                row,
                body_a,
                body_b,
                response_a,
                response_b,
                normal,
                phi,
                direction,
                row_point,
                mu,
                restitution,
                stiffness,
            )


@wp.kernel(enable_backward=False)
def free_body_headers_kernel(
    mf: MfRows,
    mf_constraint_count: wp.array[int],
    walk: WalkTrees,
    body_to_articulation: wp.array[int],
    has_target_velocity: int,
    v_hat: wp.array[float],
    rigid_body_max_depenetration_velocity: wp.array[float],
    pgs_cfm: float,
    pgs_beta: float,
    contact_w: float,
    dt: float,
    contact_speculative_scale: float,
    restitution_velocity_threshold: float,
    mf_max_constraints: int,
    weighted: int,
    material: int,
    # outputs
    mf_eff_mass_inv: wp.array2d[float],
    mf_row_w: wp.array2d[float],
):
    """The walked free-body rows' build, one thread per row slot: each row's header
    (:func:`.response_builds_rows.free_body_row_header`) from the incident velocity its walks summed
    (:func:`write_free_body_rows`) and its diagonal ``A_ii = cfm + |Z_a|^2 + |Z_b|^2``. A speed row keeps its axis
    ``e_k`` unwhitened (its law reads the body's velocity, :meth:`.response_operator.FreeBodyZ.speed`): its diagonal
    takes the walk of the covector ``e_k``, ``|L^-1 P e_k|^2``, and its incident is zero."""
    tid = wp.tid()
    world = tid // mf_max_constraints
    i = tid % mf_max_constraints
    if i >= mf_constraint_count[world]:
        return
    line = mf.first[world]
    body_a = mf.body_a[line, i]
    body_b = mf.body_b[line, i]
    diagonal = pgs_cfm
    incident = float(0.0)
    if mf.row_type[line, i] == PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT:
        if body_a >= 0:
            art = body_to_articulation[body_a]
            j0 = walk.joint_start[art]
            d0 = walk.dof_start[art]
            k = int(0)
            if mf.Z_a[line, i, 3] != 0.0:
                k = 3
            seed = walk_into(
                walk,
                j0,
                d0,
                walk.joint_end[art] - j0,
                walk.dofs[art],
                walk.dof_body[d0 + k],
                wp.spatial_vector(),
                k,
                1.0,
                v_hat,
                -1,
                mf.Z_a,
                line,
                i,
                0,
            )
            diagonal += seed[1]
    else:
        incident = mf.incident[line, i]
        if body_a >= 0:
            squares = float(0.0)
            for k in range(6):
                squares += mf.Z_a[line, i, k] * mf.Z_a[line, i, k]
            diagonal += squares
        if body_b >= 0:
            squares = float(0.0)
            for k in range(6):
                squares += mf.Z_b[line, i, k] * mf.Z_b[line, i, k]
            diagonal += squares
    inverse = float(0.0)
    if diagonal > 0.0:
        inverse = 1.0 / diagonal
    free_body_row_header(
        mf,
        line,
        i,
        incident,
        inverse,
        has_target_velocity,
        rigid_body_max_depenetration_velocity,
        pgs_beta,
        contact_w,
        dt,
        contact_speculative_scale,
        restitution_velocity_threshold,
        weighted,
        material,
        mf_eff_mass_inv,
        mf_row_w,
    )


@wp.func
def write_row(
    rows: DenseRows,
    support: RowSupport,
    line: int,
    slot: int,
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
    packed_Z: wp.array3d[float],
    row_incident: wp.array2d[float],
):
    """Row ``slot`` of line ``line``: its entries cleared, its seeds walked into them (:func:`walk_wrench`,
    :func:`walk_covector`), the slips' release on each side that slips (:func:`release_side`), and its diagonal
    ``|z|^2 + cfm``, incident velocity and rhs (:func:`.kernels.row_bias`) stored."""
    width = packed_Z.shape[2]
    for e in range(width):
        packed_Z[line, slot, e] = 0.0
    source = support.units[line, slot]
    kind = source >> UNIT_SHIFT
    index = source & ((1 << UNIT_SHIFT) - 1)
    sides = rows.sides[line, slot]
    # The seeds' incident and squares; a second seed on one side adds to the first, so its squares do not sum.
    walked = wp.vec2(0.0)
    summed = True
    if kind == UNIT_CONTACT:
        c = index // 4
        r = index - 4 * c
        body_a, body_b, normal, point, _phi, tangent0, tangent1, friction_point = contact_frame(data, law, c)
        direction, row_point = contact_row_frame(r, normal, tangent0, tangent1, point, friction_point)
        spin = r == CONTACT_SPIN_ROW
        art_a = data.art0[c]
        art_b = data.art1[c]
        shared = int(0)
        if art_a >= 0 and art_a == art_b:
            shared = 1
        walked += walk_wrench(
            walk, support, sides, body_a, art_a, 1.0, direction, row_point, spin, v_hat, 0, packed_Z, slot
        )
        walked += walk_wrench(
            walk, support, sides, body_b, art_b, -1.0, direction, row_point, spin, v_hat, shared, packed_Z, slot
        )
        summed = shared == 0
    elif kind == UNIT_LINK:
        k = index // LINK_ROWS
        row = index - LINK_ROWS * k
        direction = wp.vec3(0.0)
        direction[row % 3] = 1.0
        angular = row >= 3
        primary = link_primary[k]
        shadow = link_shadow[k]
        art_p = body_to_articulation[primary]
        art_s = body_to_articulation[shadow]
        shared = int(0)
        if art_p >= 0 and art_p == art_s:
            shared = 1
        point_p = wp.transform_point(body_q[primary], body_com[primary])
        point_s = wp.transform_point(body_q[shadow], body_com[shadow])
        walked += walk_wrench(
            walk, support, sides, primary, art_p, 1.0, direction, point_p, angular, v_hat, 0, packed_Z, slot
        )
        walked += walk_wrench(
            walk, support, sides, shadow, art_s, -1.0, direction, point_s, angular, v_hat, shared, packed_Z, slot
        )
        summed = shared == 0
    elif kind == UNIT_LIMIT or kind == UNIT_VELOCITY:
        dof = index // 2
        value = 1.0 - 2.0 * float(index - 2 * dof)
        if kind == UNIT_VELOCITY:
            value = velocity_limit_sign[index]
        walked += walk_covector(walk, support, sides, sides[0], dof, value, v_hat, 0, packed_Z, slot)
    elif kind == UNIT_MIMIC:
        walked += walk_covector(walk, support, sides, sides[0], mimic_dof0[index], 1.0, v_hat, 0, packed_Z, slot)
        walked += walk_covector(
            walk, support, sides, sides[0], mimic_dof1[index], -mimic_coef1[index], v_hat, 1, packed_Z, slot
        )
        summed = False
    # The block's first phase: the slips' release on each side that slips (the closure follows, on these rows).
    if slips_on(walk, sides[0]) != 0:
        release_side(walk, support, sides, sides[0], packed_Z, slot)
        summed = False
    if sides[1] != sides[0] and slips_on(walk, sides[1]) != 0:
        release_side(walk, support, sides, sides[1], packed_Z, slot)
        summed = False
    norm = walked[1]
    if not summed:
        norm = float(0.0)
        for e in range(width):
            z = packed_Z[line, slot, e]
            norm += z * z
    rows.diag[line, slot] = norm + rows.row_cfm[line, slot]
    row_incident[line, slot] = walked[0]
    row_bias(
        rows,
        line,
        slot,
        walked[0],
        dt,
        contact_speculative_scale,
        contact_w,
        restitution_velocity_threshold,
        weighted,
        material,
    )


@wp.func
def block_row(source: int, mimic_blocks: int) -> int:
    """Whether a row of unit source ``source`` belongs to a bilateral block: a link's, or a mimic's where mimics are
    eliminated (``enable_bilateral_preelimination``)."""
    kind = source >> UNIT_SHIFT
    if kind == UNIT_LINK:
        return 1
    if kind == UNIT_MIMIC and mimic_blocks != 0:
        return 1
    return 0


@wp.func_native(dual_native("    SYNC();\n", preludes=(lane_group("lanes"),)))
def world_sync(lanes: int):
    """Synchronize this thread's world group of ``lanes`` lanes (a power of two up to 32, within a warp; nothing on the
    CPU, where one thread runs the world)."""


def writer_lanes(rows: int, capacity: int, device) -> int:
    """A world's lane group in the one writer: the smallest power of two at or above the rows its topology fixes
    (``rows``: links, mimics and limits; contacts stride over it), at most the row capacity and a warp, so a world's
    lanes synchronize within their warp (:func:`world_sync`); one thread on the CPU.

    A world's release and closure run serially on one of its lanes, so lanes beyond its rows idle there. Measured
    (FL33D, GB300, 4,096 worlds of 6 or 12 bilateral rows, or 6 limit rows with three slips): groups of 8 and 16 ran
    the writer's phases 31-37% under the separate launches, groups of 32 and 64 (the capacity) at par to 42% over."""
    if not wp.get_device(device).is_cuda:
        return 1
    lanes = max(min(int(rows), int(capacity), 32), 1)
    return 1 << (lanes - 1).bit_length()


WRITER_THREADS = 256
"""Threads of a block of the one writer: whole world groups of up to 32 lanes."""


@cache
def walked_rows_kernel(release: bool, blocks: int, mimic_blocks: bool) -> wp.Kernel:
    """The one writer of the general placement, world-major: a world's group of ``lanes`` lanes (:func:`writer_lanes`)
    writes its rows in the block primitive's phases (FL3_design §1.4, W9), the group synchronized between them:

    1. *release* (``release``, a class whose drives can saturate): the lanes take the world's walked trees, each tree's
       slips released (:func:`.tree_factor.walk_release_tree`);
    2. *rows*: the lanes write every row of the world;
    3. *closure* (``blocks``, the largest bilateral block's rows; 0 for none): the lanes close each of the world's
       blocks (:func:`.response_builds_rows.close_block`; its rows are the links', and the mimics' with
       ``mimic_blocks``), then correct every other row on the blocks its sides touch
       (:func:`.response_builds_rows.close_row`).

    The closure's seed stays a launch after the warm start's gather (:meth:`.tree_factor.TreeFactor.seed_blocks`).
    One launch with group barriers, not a second launch for the block phase: FL33D's measurement (``fl33d/``)."""
    has_blocks = blocks > 0
    capacity = max(blocks, 1)
    mimics = int(mimic_blocks)

    def kernel(
        rows: DenseRows,
        support: RowSupport,
        counts: wp.array[int],
        lanes: int,
        data: ContactInput,
        law: PatchRowLaw,
        walk: WalkTrees,
        slips: WalkSlips,
        preelim: Preelim,
        mimic_slot: wp.array[int],
        link_slot: wp.array[int],
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
        world, lane = wp.tid()
        count = wp.min(counts[world], packed_Z.shape[1])
        line = rows.first[world]
        if wp.static(release):
            for i in range(slips.world_start[world] + lane, slips.world_start[world + 1], lanes):
                walk_release_tree(slips.world_arts[i], walk, slips, v_hat)
            world_sync(lanes)
        for slot in range(lane, count, lanes):
            write_row(
                rows, support, line, slot, data, law, walk, body_to_articulation, mimic_dof0, mimic_dof1, mimic_coef1,
                link_primary, link_shadow, velocity_limit_sign, body_q, body_com, v_hat, dt, contact_speculative_scale,
                contact_w, restitution_velocity_threshold, weighted, material, packed_Z, row_incident,
            )  # fmt: skip
        if wp.static(has_blocks):
            world_sync(lanes)
            for pe in range(preelim.world_start[world] + lane, preelim.world_start[world + 1], lanes):
                close_block(
                    pe, wp.static(capacity), line, preelim, mimic_slot, link_slot, rows.row_cfm, packed_Z,
                    rows.rhs, row_incident, rows.diag,
                )  # fmt: skip
            world_sync(lanes)
            for slot in range(lane, count, lanes):
                if block_row(support.units[line, slot], wp.static(mimics)) == 0:
                    close_row(
                        line, slot, rows.sides[line, slot], wp.static(capacity), preelim, rows.row_cfm, packed_Z,
                        rows.rhs, rows.diag,
                    )  # fmt: skip

    name = "write_walked_rows" + ("_release" if release else "") + (f"_blocks{blocks}" if has_blocks else "")
    name += "_mimics" if mimic_blocks else ""
    return python_kernel(kernel, name, launch_bounds=(WRITER_THREADS, 3))


def writer_kernel(solver) -> wp.Kernel:
    """The one writer's variant for the solver's class (:func:`walked_rows_kernel`), from its topology: the release
    phase where a drive can saturate, the block phase where bilateral blocks exist (mimics among them under
    ``enable_bilateral_preelimination``)."""
    factor = solver.factor
    blocks = factor.block_capacity if factor.eliminates else 0
    return walked_rows_kernel(
        factor.walk_slip_capacity > 0, blocks, bool(blocks and solver.options.enable_bilateral_preelimination)
    )

# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The one writer of the general placement (FL3.3) and of the sparse diagonal (FL3.5): every row a seed and a law,
whitened by the walk.

A dense row of the general placement (case K6) holds its unit source (:func:`.dense_rows.unit_source`), which its
producer records with the row's slot and records, in place of a Jacobian. After the row pass, the writer takes every
row of a world on the world's lanes (:func:`walked_rows_kernel`):

- **the seed**, from the row's law (:data:`.dense_rows.UNIT_CONTACT` and the other kinds): a contact row's wrench on
  each of its bodies, ``(d, (p - o) x d)`` at the body's frame ``o`` (the factor's; its spin row's ``(0, n)``),
  the second body's negated; a link row's unit wrench at its two pieces' centres of mass, the shadow's negated; a joint
  limit's and a velocity limit's covector ``+-e_c``, a mimic's ``e_f - c e_l``;
- **the walk** (:func:`.tree_factor.walk_paths_prelude`): each seed walked up its root path into the packed sink at its
  side's entries, stored, a second seed on one tree adding where their paths meet. A wrench enters as the step's ``J``,
  the finish's motion columns shifted to the walk's frames (:attr:`.tree_factor.WalkTrees.motion`), and the refresh's
  factor whitens it, so a step between
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

**The support placement (case K1, FL3.4)** writes the same rows into its support sink (:func:`support_rows_kernel`):
a world's lane group strides over its units, a lane walks a unit's rows together (a contact's 1, 3 or 4, a joint row's
1) through the class's path table in constant memory, and stores each footprint at its rank in the row's support
(factor coordinates), the unit's Gram, its records and its header from the same walk. A slipping drive's ``K`` leaves
the rows and the leave by its release along its path (the predictor's border, :func:`.tree_kinetic_state.border_solve`): the block primitive with
the path's coordinate basis, which keeps every row inside its support. Its sweep's change goes to the step's one leave
(:func:`.tree_factor.walk_leave_kernel`).
"""

from __future__ import annotations

from functools import cache

import warp as wp

from . import field_group
from .contact_input import ContactInput, contact_frame
from .contact_laws import CONTACT_SPIN_ROW, contact_row_entry, contact_row_frame, contact_seed, spin_seed
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
)
from .device_preludes import DevicePrelude, dual_native, lane_group, substitute
from .field_group import Field
from .friction_patches import PatchRowLaw
from .kernel_template import python_kernel
from .kernels import PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT, free_body_contact, free_body_contact_records, row_bias
from .projection_laws import friction_pair_metric
from .response_builds_rows import close_block, close_row, free_body_row_header
from .row_build import write_contact_row
from .row_laws import LINK_ROWS
from .solver_fields import Preelim
from .sparse_mass_matrix import _SparseRowSupport
from .tree_factor import (
    WALK_COORDINATE,
    WALK_FRAME,
    WALK_LINK,
    WalkRelease,
    WalkTrees,
    release_path,
    walk_covector_into,
    walk_into,
    walk_prelude,
)

__all__ = [
    "SlotRows",
    "SupportSink",
    "SupportUnit",
    "bind_slot_rows",
    "free_body_headers_kernel",
    "support_rows_kernel",
    "walked_rows_kernel",
    "write_free_body_rows",
    "write_slot_rows",
    "writer_kernel",
    "writer_lanes",
]


@wp.func
def walk_wrench(
    walk: WalkTrees,
    support: RowSupport,
    sides: wp.vec2i,
    group: int,
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
    it), into its side's entries of row ``slot`` (:func:`.dense_rows.jacobian_row`: the packed rows, or the size group
    ``group``'s): stored, or added with ``add`` (a second seed on the same side). Return its part of the incident velocity ``J v_hat`` (the step's ``J`` entries the walk passes) and the sum of its
    footprints' squares. A body without response coordinates adds nothing."""
    if body < 0 or art < 0:
        return wp.vec2(0.0)
    if support.dofs[art] <= 0:
        return wp.vec2(0.0)
    at = jacobian_row(support, group, art, sides)
    wrench = body_wrench(walk, body, sign * direction, point, torque)
    return walk_body(walk, art, body, wrench, v_hat, add, packed_Z, at[0], slot, at[1])


@wp.func
def body_wrench(walk: WalkTrees, body: int, direction: wp.vec3, point: wp.vec3, torque: bool) -> wp.spatial_vector:
    """The seed of a row acting on ``body`` along ``direction`` at ``point`` (a pure ``torque`` about it), by the
    contact law (:func:`.contact_laws.contact_seed`, :func:`.contact_laws.spin_seed`): a wrench at the body's frame, the
    factor's (:attr:`.tree_factor.WalkTrees.link`), where the walk shifts the finish's motion columns to take the step's
    ``J``. The lever arm is the world points' difference here; the law takes it from wherever it is formed."""
    if torque:
        return spin_seed(direction)
    j = walk.body_joint[body]
    frame = wp.vec3(walk.link[j, 4], walk.link[j, 5], walk.link[j, 6])
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
        art,
        j0,
        walk.dof_start[art],
        walk.joint_end[art] - j0,
        walk.dofs[art],
        walk.body_joint[body] - j0,
        wrench,
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
    group: int,
    art: int,
    dof: int,
    value: float,
    v_hat: wp.array[float],
    add: int,
    packed_Z: wp.array3d[float],
    slot: int,
) -> wp.vec2:
    """Walk a row's covector ``value e_dof`` (a joint row's seed) into its side's entries of row ``slot`` (in the size
    group ``group``'s rows where they are not packed; added with ``add``); return its part of the incident velocity,
    ``value v_hat[dof]``, and the sum of its footprints' squares."""
    if art < 0:
        return wp.vec2(0.0)
    if support.dofs[art] <= 0:
        return wp.vec2(0.0)
    d0 = walk.dof_start[art]
    j0 = walk.joint_start[art]
    at = jacobian_row(support, group, art, sides)
    return walk_covector_into(
        walk,
        j0,
        d0,
        walk.joint_end[art] - j0,
        walk.dofs[art],
        walk.dof_body[dof],
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
    walk: WalkTrees, support: RowSupport, sides: wp.vec2i, group: int, art: int, packed_Z: wp.array3d[float], slot: int
):
    """The slips' release on articulation ``art``'s side of row ``slot`` (in the size group ``group``'s rows where they
    are not packed): ``z <- R z``, the root-most slip first, each along its path leaf first
    (the predictor's border, :func:`.tree_kinetic_state.border_solve`), the decided operator's row on the all-``K`` walk. The side holds the whole
    tree, and ``R`` keeps every row's support."""
    if art < 0:
        return
    at = jacobian_row(support, group, art, sides)
    for j in range(walk.release.count[art]):
        release_path(walk.release, art, j, packed_Z, at[0], slot, at[1])


# Two blocks of 256 a multiprocessor (128 registers): the walk's one-row seeds with their held step sit at the bound.
@wp.kernel(enable_backward=False, launch_bounds=(256, 2))
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
            seed = walk_covector_into(
                walk,
                j0,
                d0,
                walk.joint_end[art] - j0,
                walk.dofs[art],
                walk.dof_body[d0 + k],
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
        # The row's records by the contact law (the producer's, folded into the writer), then its seeds.
        body_a, body_b, direction, row_point = write_contact_row(data, law, c, r, False)
        sides = rows.sides[line, slot]
        spin = r == CONTACT_SPIN_ROW
        art_a = data.art0[c]
        art_b = data.art1[c]
        shared = int(0)
        if art_a >= 0 and art_a == art_b:
            shared = 1
        walked += walk_wrench(
            walk, support, sides, 0, body_a, art_a, 1.0, direction, row_point, spin, v_hat, 0, packed_Z, slot
        )
        walked += walk_wrench(
            walk, support, sides, 0, body_b, art_b, -1.0, direction, row_point, spin, v_hat, shared, packed_Z, slot
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
            walk, support, sides, 0, primary, art_p, 1.0, direction, point_p, angular, v_hat, 0, packed_Z, slot
        )
        walked += walk_wrench(
            walk, support, sides, 0, shadow, art_s, -1.0, direction, point_s, angular, v_hat, shared, packed_Z, slot
        )
        summed = shared == 0
    elif kind == UNIT_LIMIT or kind == UNIT_VELOCITY:
        dof = index // 2
        value = 1.0 - 2.0 * float(index - 2 * dof)
        if kind == UNIT_VELOCITY:
            value = velocity_limit_sign[index]
        walked += walk_covector(walk, support, sides, 0, sides[0], dof, value, v_hat, 0, packed_Z, slot)
    elif kind == UNIT_MIMIC:
        walked += walk_covector(walk, support, sides, 0, sides[0], mimic_dof0[index], 1.0, v_hat, 0, packed_Z, slot)
        walked += walk_covector(
            walk, support, sides, 0, sides[0], mimic_dof1[index], -mimic_coef1[index], v_hat, 1, packed_Z, slot
        )
        summed = False
    # The block's first phase: the slips' release on each side that slips (the closure follows, on these rows).
    if slips_on(walk, sides[0]) != 0:
        release_side(walk, support, sides, 0, sides[0], packed_Z, slot)
        summed = False
    if sides[1] != sides[0] and slips_on(walk, sides[1]) != 0:
        release_side(walk, support, sides, 0, sides[1], packed_Z, slot)
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
def walked_rows_kernel(blocks: int, mimic_blocks: bool) -> wp.Kernel:
    """The one writer of the general placement, world-major: a world's group of ``lanes`` lanes (:func:`writer_lanes`)
    writes its rows in the block primitive's phases (FL3_design §1.4, W9), the group synchronized between them. The
    slips' release ``R`` (the first phase) is the predictor's border's (:func:`.tree_kinetic_state.border_solve`), which
    each row takes on the sides that slip (:func:`release_side`):

    1. *rows*: the lanes write every row of the world, except a world an in-kernel owner holds (``owned``: the local
       owners' packets walk their own seeds);
    2. *closure* (``blocks``, the largest bilateral block's rows; 0 for none): the lanes close each of the world's
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
        owner: wp.array[int],
        owned: int,
        lanes: int,
        data: ContactInput,
        law: PatchRowLaw,
        walk: WalkTrees,
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
        # A world an in-kernel owner holds (``owned`` and ``owner`` nonzero: the local owners' packets, whose kernels
        # walk their seeds) is not the writer's: its whole group returns before any barrier.
        if owned != 0 and owner[world] != 0:
            return
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

    name = "write_walked_rows" + (f"_blocks{blocks}" if has_blocks else "")
    name += "_mimics" if mimic_blocks else ""
    return python_kernel(kernel, name, launch_bounds=(WRITER_THREADS, 3))


def writer_kernel(solver) -> wp.Kernel:
    """The one writer's variant for the solver's class (:func:`walked_rows_kernel`), from its topology: the block phase
    where bilateral blocks exist (mimics among them under ``enable_bilateral_preelimination``)."""
    factor = solver.factor
    blocks = factor.block_capacity if factor.eliminates else 0
    return walked_rows_kernel(blocks, bool(blocks and solver.options.enable_bilateral_preelimination))


# ---------------------------------------------------------------------------------------------------------------------
# The support placement (case K1, FL3.4): the support sink, a lane per unit, the release and the leave
# ---------------------------------------------------------------------------------------------------------------------


_SUPPORT_TABLE = """    static __constant__ int SUPPORT_NODE[$n] = {$node};"""
_SUPPORT_TABLE_CPU = """    static const int SUPPORT_NODE[$n] = {$node};"""

_SUPPORT = """    // One unit's rows into the support sink (walk_rows.support_rows_kernel): its seeds walked through the class's
    // path table (WALK_UP, WALK_SEED), each footprint stored at its rank in the row's support (SUPPORT_NODE: a
    // coordinate's factor coordinate, the support mask's bit), the unit's Gram and incident from the same walk.
    const auto POPC = [](unsigned long long bits) -> int {
#if defined(__CUDA_ARCH__)
        return __popcll(bits);
#else
        return __builtin_popcountll(bits);
#endif
    };
    const auto LOW = [](unsigned long long bits) -> int {
#if defined(__CUDA_ARCH__)
        return __ffsll(bits) - 1;
#else
        return __builtin_ctzll(bits);
#endif
    };
    const auto& walk = sink.walk;
    const auto& release = sink.release;
    const auto& support = sink.support;
    const int rows = unit.rows, art_a = unit.art_a, art_b = unit.art_b, body_a = unit.body_a, body_b = unit.body_b;
    const int dof0 = unit.dof0, dof1 = unit.dof1;
    const int S = sink.row_factor.shape[2];
    const int row0 = unit.line * sink.row_factor.shape[1] + unit.slot;
    float* const Z = sink.row_factor.data + (size_t)row0 * S;
    const bool covector = dof0 >= 0;
    float incident[$rows], norm[$rows], zc[$rows], cross[6];
    for (int r = 0; r < $rows; ++r) incident[r] = norm[r] = zc[r] = 0.0f;
    for (int k = 0; k < 6; ++k) cross[k] = 0.0f;
    const auto gram = [&]() {
        if ($rows >= 3) {
            cross[0] += zc[0] * zc[1];
            cross[1] += zc[0] * zc[2];
            cross[2] += zc[1] * zc[2];
        }
        if ($rows >= 4) {
            cross[3] += zc[0] * zc[$last];
            cross[4] += zc[1] * zc[$last];
            cross[5] += zc[2] * zc[$last];
        }
    };
    unsigned long long masks[2] = {0ull, 0ull};
    int bases[2] = {0, 0};
    int count = 0;
    bool reread = false;
    for (int side = 0; side < 2; ++side) {
        const int art = side == 0 ? art_a : art_b;
        if (art < 0 || (side == 1 && art == art_a) || sink.response_dofs.data[art] <= 0) continue;
        const int j0 = walk.joint_start.data[art], d0 = walk.dof_start.data[art];
        unsigned long long mask = 0ull;
        int seeds = 0;
        if (covector) {
            mask = sink.dof_mask.data[dof0 - d0];
            seeds = 1;
            if (dof1 >= 0) {
                mask |= sink.dof_mask.data[dof1 - d0];
                seeds = 2;
            }
        } else {
            if (body_a >= 0 && art_a == art) {
                mask |= sink.body_mask.data[body_a];
                ++seeds;
            }
            if (body_b >= 0 && art_b == art) {
                mask |= sink.body_mask.data[body_b];
                ++seeds;
            }
        }
        if (mask == 0ull) continue;
        const int base = count;
        masks[side] = mask;
        bases[side] = base;
        count += POPC(mask);
        // Two seeds on one tree add where their paths meet: their side is cleared, then re-read for the Gram.
        const bool add = seeds > 1;
        if (add) {
            reread = true;
            for (int r = 0; r < rows; ++r)
                for (int e = base; e < count; ++e) Z[r * S + e] = 0.0f;
        }
        const float* data = walk.data.data + (size_t)d0 * $coordinate;
        const float* link = walk.link.data + (size_t)j0 * $link;
        const float* now = reinterpret_cast<const float*>(walk.motion.data + d0);
        const float* origin = reinterpret_cast<const float*>(walk.origin.data + art);
        const float* vh = sink.v_hat.data + d0;
        const auto rank = [&](int c) -> int { return base + POPC(mask & ((1ull << SUPPORT_NODE[c]) - 1ull)); };
        const auto store = [&](int r, int c, float z, float) {
            if (r >= rows) return;
            float* entry = Z + r * S + rank(c);
            if (add) {
                *entry += z;
                return;
            }
            *entry = z;
            norm[r] += z * z;
            zc[r] = z;
            if (r == $last) gram();
        };
        float p[6 * $rows], w[6 * $rows];
        if (covector) {
            for (int k = 0; k < seeds; ++k) {
                const int c = (k == 0 ? dof0 : dof1) - d0;
                float x[$rows];
                for (int r = 0; r < $rows; ++r) x[r] = 0.0f;
                x[0] = k == 0 ? unit.value0 : unit.value1;
                // A covector enters at its coordinate: its body's coordinates after it (below it in factor order) are
                // the row's zeros.
                if (!add)
                    for (int e = base; e < rank(c); ++e) Z[e] = 0.0f;
                for (int i = 0; i < 6 * $rows; ++i) p[i] = w[i] = 0.0f;
                WALK_SEED(data, c, x, p, store);
                WALK_UP(data, link, nullptr, nullptr, walk.dof_body.data[d0 + c], c - 1, p, w, store);
                incident[0] += x[0] * vh[c];
            }
        } else {
            for (int k = 0; k < 2; ++k) {
                const int body = k == 0 ? body_a : body_b;
                if (body < 0 || (k == 0 ? art_a : art_b) != art) continue;
                const float sign = k == 0 ? 1.0f : -1.0f;
                const int j = walk.body_joint.data[body];
                const int local = j - j0;
                // The seeds' wrenches at the body's frame (the factor's), where the walk's shifted columns are J's.
                const float* o = walk.link.data + (size_t)j * $link + $frame;
                for (int r = 0; r < $rows; ++r) {
                    float* q = p + 6 * r;
                    const float* d = unit.frame.data[r == 1 || r == 2 ? r : 0];
                    const auto& at = r == 0 ? unit.point : unit.friction_point;
                    const float f[3] = {sign * d[0], sign * d[1], sign * d[2]};
                    const float l[3] = {at[0] - o[0], at[1] - o[1], at[2] - o[2]};
                    if (r >= rows) {
                        for (int i = 0; i < 6; ++i) q[i] = 0.0f;
                    } else if ($spin && r == 3) {
                        q[0] = q[1] = q[2] = 0.0f;
                        q[3] = f[0];
                        q[4] = f[1];
                        q[5] = f[2];
                    } else {
                        q[0] = f[0];
                        q[1] = f[1];
                        q[2] = f[2];
                        q[3] = l[1] * f[2] - l[2] * f[1];
                        q[4] = l[2] * f[0] - l[0] * f[2];
                        q[5] = l[0] * f[1] - l[1] * f[0];
                    }
                    for (int i = 0; i < 6; ++i) w[6 * r + i] = q[i];
                }
                // The step's J entry of each footprint against v_hat: the rows' incident velocity.
                const auto sink = [&](int r, int c, float z, float j) {
                    if (r < rows) incident[r] += j * vh[c];
                    store(r, c, z, j);
                };
                WALK_UP(data, link, now, origin, local, WALK_FIRST_OF(local + 1) - 1, p, w, sink);
            }
        }
        // A slipping tree's release (the predictor's border), root-most slip first, on the side's entries along each
        // slip's path: entries off the row's support stay zero, so the side keeps its support.
        const int slips = release.count.data[art];
        if (slips > 0) {
            reread = true;
            const int cap = release.dof.shape[1], width = release.b.shape[2];
            for (int j = 0; j < slips; ++j) {
                const int i = release.dof.data[art * cap + j];
                const size_t at = ((size_t)art * cap + j) * width;
                const float *b = release.b.data + at, *a = release.a.data + at, *g = release.g.data + at;
                float s[$rows];
                for (int r = 0; r < $rows; ++r) s[r] = 0.0f;
                // The path, leaf first (step t); a coordinate off the row's support holds zero and leaves s at zero.
                for (int body = walk.dof_body.data[d0 + i], c = i, t = 0; body >= 0;) {
                    for (; c >= WALK_FIRST_OF(body); --c, ++t) {
                        if (((mask >> SUPPORT_NODE[c]) & 1ull) == 0ull) continue;
                        float* entry = Z + rank(c);
                        for (int r = 0; r < rows; ++r) {
                            const float u = entry[r * S] - b[t] * s[r];
                            s[r] += a[t] * u;
                            entry[r * S] = g[t] * u;
                        }
                    }
                    body = WALK_PARENT_OF(body);
                    if (body >= 0) c = WALK_FIRST_OF(body + 1) - 1;
                }
            }
        }
    }
    // The pad past the support; the rows' coordinates (explicit) or their support class (the table).
    for (int r = 0; r < rows; ++r)
        for (int e = count; e < S; ++e) Z[r * S + e] = 0.0f;
    if (support.enabled == 0) {
        int* const D = sink.row_dof.data + (size_t)row0 * S;
        for (int side = 0; side < 2; ++side) {
            if (masks[side] == 0ull) continue;
            int e = bases[side];
            const int offset = sink.world_offset.data[side == 0 ? art_a : art_b];
            for (unsigned long long bits = masks[side]; bits != 0ull; bits &= bits - 1ull, ++e)
                for (int r = 0; r < rows; ++r) D[r * S + e] = offset + LOW(bits);
        }
        for (int r = 0; r < rows; ++r)
            for (int e = count; e < S; ++e) D[r * S + e] = -1;
    } else {
        int cls = 0;
        if (covector) {
            cls = support.limit_class.data[dof0 - walk.dof_start.data[art_a]];
        } else {
            const auto side_of = [&](int body, int art) {
                return body >= 0 && art >= 0 && sink.response_dofs.data[art] > 0 && sink.body_mask.data[body] != 0ull;
            };
            const bool a = side_of(body_a, art_a), b = side_of(body_b, art_b);
            const int class_a = a ? support.body_class.data[body_a] : 0;
            const int class_b = b ? support.body_class.data[body_b] : 0;
            cls = support.pair_class.data[class_a * support.pair_class.shape[1] + class_b];
        }
        for (int r = 0; r < rows; ++r) support.row_class.data[row0 + r] = cls;
    }
    if (reread) {
        for (int r = 0; r < $rows; ++r) norm[r] = zc[r] = 0.0f;
        for (int k = 0; k < 6; ++k) cross[k] = 0.0f;
        for (int e = 0; e < count; ++e) {
            for (int r = 0; r < rows; ++r) {
                zc[r] = Z[r * S + e];
                norm[r] += zc[r] * zc[r];
            }
            if (rows == $rows) gram();
        }
    }
    for (int r = 0; r < rows; ++r) {
        sink.diag.data[row0 + r] = norm[r] + sink.row_cfm.data[row0 + r];
        sink.row_incident.data[row0 + r] = incident[r];
    }
$record"""
"""The support sink's unit walker, ``$rows`` rows a lane (:func:`support_rows_kernel`)."""

_SUPPORT_RECORD = """    // A full unit's block record from its Gram (sparse_contact.CONTACT_RECORD_FLOATS): its couplings W_nt1, W_nt2,
    // W_t1t2 and 1 / W_nn, then the spin row's couplings W_ns, W_t1s, W_t2s (contact torsion) or, written by the
    // kernel from these, the tangent pair's metric (projection_laws.friction_pair_metric).
    if (rows == $rows) {
        float* record = sink.records.data + ((size_t)unit.line * sink.records.shape[1] + unit.slot / 3)
                        * sink.records.shape[2];
        record[0] = cross[0];
        record[1] = cross[1];
        record[2] = cross[2];
        record[3] = 1.0f / (norm[0] + sink.row_cfm.data[row0]);
$tail
    }
"""
_SPIN_TAIL = """        record[4] = cross[3];
        record[5] = cross[4];
        record[6] = cross[5];"""

@cache
def _support_table(node: tuple[int, ...]) -> DevicePrelude:
    """The class's factor coordinate of each tree coordinate, in constant memory: the support mask's bit."""
    values = {"n": len(node), "node": ", ".join(str(int(k)) for k in node)}
    return DevicePrelude(
        f"support_table_{len(node)}", substitute(_SUPPORT_TABLE, values), cpu=substitute(_SUPPORT_TABLE_CPU, values)
    )


def _support_snippet(parent: tuple, first: tuple, node: tuple, rows: int, records: bool, spin: bool) -> str:
    record = ""
    if records:
        record = substitute(_SUPPORT_RECORD, rows=rows, tail=_SPIN_TAIL if spin else "")
    body = substitute(
        _SUPPORT,
        rows=rows,
        last=rows - 1,
        coordinate=WALK_COORDINATE,
        link=WALK_LINK,
        frame=WALK_FRAME,
        spin=int(spin),
        record=record,
    )
    return dual_native(body, preludes=(walk_prelude(parent, first, rows), _support_table(node)))


@wp.struct
class SupportSink:
    """What the support placement's writer reads and writes besides a unit's seeds (:func:`support_rows_kernel`): the
    walk and its release, the support table, each body's and coordinate's support mask (factor coordinates), each
    articulation's first world coordinate and response coordinates, the predictor ``v_hat``, and the rows' ``cfm``,
    entries, coordinates, incident velocities, diagonals and block records."""

    walk: WalkTrees
    release: WalkRelease
    support: _SparseRowSupport
    body_mask: wp.array[wp.uint64]
    dof_mask: wp.array[wp.uint64]
    world_offset: wp.array[int]
    response_dofs: wp.array[int]
    v_hat: wp.array[float]
    row_cfm: wp.array2d[float]
    row_factor: wp.array3d[float]
    row_dof: wp.array3d[int]
    row_incident: wp.array2d[float]
    diag: wp.array2d[float]
    records: wp.array3d[float]


@wp.struct
class SupportUnit:
    """One unit's rows (``line``, ``slot``, ``rows``) and seeds: a contact's two bodies and their articulations with its
    frame's rows (the normal, then the tangents) at its ``point`` and ``friction_point``, or a joint row's covectors
    ``value e_dof`` (``dof0 >= 0``; a mimic's second at ``dof1``) on articulation ``art_a``."""

    line: int
    slot: int
    rows: int
    art_a: int
    body_a: int
    art_b: int
    body_b: int
    frame: wp.mat33
    point: wp.vec3
    friction_point: wp.vec3
    dof0: int
    value0: float
    dof1: int
    value1: float


@cache
def support_rows_kernel(
    parent: tuple[int, ...], first: tuple[int, ...], node: tuple[int, ...], *, records: bool, torsion: bool, lanes: int
) -> wp.Kernel:
    """The support placement's one writer (case K1): ``dim = (worlds, lanes)``, a world's lane group of ``lanes``
    striding over its rows: first each contact row's records by the contact law (a lane a row: the producer's work,
    folded), then a lane per unit (a contact's rows from its first, a joint row alone). The generator form
    (Gamma per body, ``z = Gamma w``) measured slower on g1 than this direct walk, so the writer walks each unit.

    The class's tree (``parent``, ``first``: :func:`.tree_factor.walk_prelude`'s table) and each coordinate's factor
    coordinate ``node`` (the support masks' bit) are baked into constant memory. A contact unit of three rows (four
    under contact ``torsion``) is walked together and its Gram writes its block record where ``records`` (the
    triplet sweeps'); a unit of one row (a normal alone, a limit, a velocity limit, a mimic) by the one-row walker. Each
    contact row's records are written here by the contact law, its unit source where its slot is reserved
    (:func:`.kernels.allocate_world_contact_slots`); a joint row's records and unit source are its producer's (the
    limit, velocity-limit and mimic producers); the writer stores its footprints at their ranks in its support (``row_factor``, with the
    coordinates ``row_dof`` or the support class ``row_class``), its diagonal ``|z|^2 + cfm``, its incident ``J v_hat``
    and its rhs by the row law (:func:`.kernels.row_bias`).
    """
    unit_rows = 4 if torsion else 3

    def walker(rows: int, with_records: bool):
        """The unit walker of ``rows`` rows (:data:`_SUPPORT`): a native function of its own name."""

        def native(sink: SupportSink, unit: SupportUnit): ...

        native.__name__ = native.__qualname__ = f"support_walk_{rows}"
        return wp.func_native(_support_snippet(parent, first, node, rows, with_records, torsion and rows == 4))(native)

    walk_one = walker(1, False)
    walk_unit = walker(unit_rows, records)
    def kernel(
        rows: DenseRows,
        sources: wp.array2d[int],
        counts: wp.array[int],
        data: ContactInput,
        law: PatchRowLaw,
        sink: SupportSink,
        mimic_dof0: wp.array[int],
        mimic_dof1: wp.array[int],
        mimic_coef1: wp.array[float],
        velocity_limit_sign: wp.array[float],
        dt: float,
        contact_speculative_scale: float,
        contact_w: float,
        restitution_velocity_threshold: float,
        weighted: int,
        material: int,
    ):
        world, lane = wp.tid()
        line = rows.first[world]
        count = wp.min(counts[world], sink.row_factor.shape[1])
        # The contact rows' records by the contact law, a lane a row (the producer's work, folded into the writer),
        # before any row's header reads them.
        for slot in range(lane, count, wp.static(lanes)):
            source = sources[line, slot]
            if source >> UNIT_SHIFT == UNIT_CONTACT:
                index = source & ((1 << UNIT_SHIFT) - 1)
                c = index // 4
                write_contact_row(data, law, c, index - 4 * c, False)
        world_sync(wp.static(lanes))
        # A lane a unit: a contact's rows from its first, a joint row alone.
        for slot in range(lane, count, wp.static(lanes)):
            source = sources[line, slot]
            kind = source >> UNIT_SHIFT
            index = source & ((1 << UNIT_SHIFT) - 1)
            unit = SupportUnit()
            unit.line = line
            unit.slot = slot
            unit.rows = 1
            unit.art_b = -1
            unit.body_a = -1
            unit.body_b = -1
            unit.dof0 = -1
            unit.dof1 = -1
            if kind == UNIT_CONTACT:
                c = index // 4
                if index - 4 * c != 0:
                    continue  # the unit's first row's lane writes it
                unit.rows = wp.min(data.slots_needed[c], wp.static(unit_rows))
                body_a, body_b, normal, point, _phi, tangent0, tangent1, friction_point = contact_frame(data, law, c)
                unit.art_a = data.art0[c]
                unit.body_a = body_a
                unit.art_b = data.art1[c]
                unit.body_b = body_b
                unit.frame = wp.matrix_from_rows(normal, tangent0, tangent1)
                unit.point = point
                unit.friction_point = friction_point
            elif kind == UNIT_LIMIT or kind == UNIT_VELOCITY or kind == UNIT_MIMIC:
                unit.art_a = rows.sides[line, slot][0]
                unit.dof0 = index // 2
                unit.value0 = 1.0 - 2.0 * float(index - 2 * unit.dof0)
                if kind == UNIT_VELOCITY:
                    unit.value0 = velocity_limit_sign[index]
                if kind == UNIT_MIMIC:
                    unit.dof0 = mimic_dof0[index]
                    unit.value0 = 1.0
                    unit.dof1 = mimic_dof1[index]
                    unit.value1 = -mimic_coef1[index]
            else:
                continue
            if unit.rows == 1:
                walk_one(sink, unit)
            else:
                walk_unit(sink, unit)
                if wp.static(records and not torsion):
                    if unit.rows == 3:
                        # The friction pair's metric of the diagonals as written, which the triplet sweeps reuse.
                        k = slot // 3
                        coupling = sink.records[line, k, 2]
                        _scale, largest, smallest, axis = friction_pair_metric(
                            sink.diag[line, slot + 1], coupling, sink.diag[line, slot + 2]
                        )
                        sink.records[line, k, 4] = largest
                        sink.records[line, k, 5] = smallest
                        sink.records[line, k, 6] = axis[0]
                        sink.records[line, k, 7] = axis[1]
            for r in range(unit.rows):
                incident = sink.row_incident[line, slot + r]
                row_bias(
                    rows, line, slot + r, incident, dt, contact_speculative_scale, contact_w,
                    restitution_velocity_threshold, weighted, material,
                )  # fmt: skip

    name = f"support_rows_{len(node)}_{abs(hash((parent, first, node))) % (1 << 32):08x}"
    name += ("_records" if records else "") + ("_torsion" if torsion else "") + f"_{lanes}"
    # Two blocks of 256 a multiprocessor within the register file (128 registers): the four-row walker holds 166
    # unbounded, and the bound costs it 16 B of stack (sm_120, ptxas).
    return python_kernel(kernel, name, launch_bounds=(256, 2))


# ---------------------------------------------------------------------------------------------------------------------
# The sparse diagonal (case K2): a tree beside a star, its rows into the slots sink (FL3.5)
# ---------------------------------------------------------------------------------------------------------------------
#
# A world of the sparse diagonal pairs a dense arm (its tree walks) with a star of one-coordinate keys on a fixed root.
# A row's seed on the arm walks into the row's dense part, the arm's coordinates in the walk's order; a seed on a key is
# one step to the world, z = (s . w) / ell, the key's step J entry over the root of its pivot (the J / sqrt(m) slot,
# 1 / ell = sqrt(m^-1) of the factor's diagonal), into the row's two slots. The sweep reads the dense part and the slots
# (:class:`.response_operator.SparsePairZ`) and leaves the arm by the walk (:func:`.tree_factor.walk_leave_kernel`), the
# release its rows carry.

SLOT_ROWS = (
    Field("single_dof", wp.int32, ("body",), owner="forest", attr="body_single_response_dof"),
    Field("dof_offset", wp.int32, ("articulation",), owner="forest", attr="articulation_world_dof_offset"),
    Field("inverse_mass", wp.float32, ("dof",), owner="factor", attr="diagonal_inverse_mass"),
    Field("star", wp.int32, attr="sparse_diagonal_response_size", owner="placement"),
    Field("dense_group", wp.int32, ("world",), attr="sparse_diagonal_dense_groups", owner="placement"),
    Field("Z", wp.float32, ("group", "row", "dense"), owner="dense"),
    Field("dof", wp.int32, ("world", "row", 2), attr="sparse_diagonal_row_dof", owner="placement"),
    Field("z", wp.float32, ("world", "row", 2), attr="sparse_diagonal_row_z", owner="placement"),
    Field("incident", wp.float32, ("world", "row"), attr="row_incident", owner="placement"),
)
"""The slots sink of the sparse diagonal: each body's single response coordinate, the articulations' first world
coordinates, the factor's diagonal inverse mass (a key's ``1 / ell^2``), the star's coordinates (``star``), each world's
dense group; the rows' dense parts by group (``Z``, ``J_by_size`` of the arm's size), their two slots (a world
coordinate, -1 for none, and its footprint) and their incident velocities."""

SlotRows = field_group.struct("SlotRows", SLOT_ROWS, __name__)


def bind_slot_rows(solver) -> SlotRows:
    """The sparse diagonal's slots sink (:data:`SLOT_ROWS`) of ``solver``."""
    placement = solver.placement
    return field_group.bind(
        SlotRows,
        placement=placement,
        forest=solver.forest,
        factor=solver.factor,
        Z=placement.J_by_size[placement.sparse_diagonal_dense_size],
    )


@wp.func
def star_step(
    slots: SlotRows,
    data: ContactInput,
    v_hat: wp.array[float],
    art: int,
    body: int,
    sign: float,
    row: int,
    direction: wp.vec3,
    point: wp.vec3,
):
    """A seed on the star, one step to the world: the key's world coordinate and footprint ``z = (s . w) / ell``, ``s .
    w`` the step's ``J`` entry of ``sign``'s contact row ``row`` along ``direction`` at ``point``
    (:func:`.contact_laws.contact_row_entry`) and ``1 / ell = sqrt(m^-1)``, and its incident velocity ``j v_hat``;
    ``(-1, 0, 0)`` for a body without a single response coordinate in its articulation's window."""
    coordinate = int(-1)
    value = float(0.0)
    incident = float(0.0)
    dof = slots.single_dof[body]
    if dof >= 0:
        local = dof - data.dof_start[art]
        if local >= 0 and local < slots.star:
            coordinate = slots.dof_offset[art] + local
            j = contact_row_entry(row, data.motion[dof], direction, point, data.origin[art]) * sign
            value = j * wp.sqrt(slots.inverse_mass[dof])
            incident = j * v_hat[dof]
    return coordinate, value, incident


@wp.func
def slot_seed(
    walk: WalkTrees,
    support: RowSupport,
    slots: SlotRows,
    data: ContactInput,
    v_hat: wp.array[float],
    sides: wp.vec2i,
    group: int,
    body: int,
    art: int,
    sign: float,
    row: int,
    direction: wp.vec3,
    point: wp.vec3,
    add: int,
    slot: int,
):
    """One endpoint of contact row ``row`` (``slot``): on a key, its slot (:func:`star_step`); on the arm, walked into
    the row's dense part (:func:`walk_wrench`, the spin row a torque about the normal; stored, or added with ``add``).
    Returns the key's coordinate (-1 for none), its footprint, the seed's incident velocity and its footprints' sum of
    squares on the arm."""
    coordinate = int(-1)
    value = float(0.0)
    walked = wp.vec2(0.0)
    if body >= 0 and art >= 0:
        if data.response_dofs[art] == slots.star:
            coordinate, value, incident = star_step(slots, data, v_hat, art, body, sign, row, direction, point)
            walked = wp.vec2(incident, 0.0)
        else:
            walked = walk_wrench(
                walk,
                support,
                sides,
                group,
                body,
                art,
                sign,
                direction,
                point,
                row == CONTACT_SPIN_ROW,
                v_hat,
                add,
                slots.Z,
                slot,
            )
    return coordinate, value, walked


@wp.func
def contact_seeds(
    walk: WalkTrees,
    support: RowSupport,
    slots: SlotRows,
    data: ContactInput,
    v_hat: wp.array[float],
    sides: wp.vec2i,
    group: int,
    slot: int,
    row: int,
    direction: wp.vec3,
    point: wp.vec3,
    body_a: int,
    art_a: int,
    body_b: int,
    art_b: int,
):
    """Contact row ``row``'s two endpoints (:func:`slot_seed`), the second's seed negated: the row's incident velocity and
    its arm footprints' sum of squares (``x``, ``y``; negative where both endpoints walk the arm, whose footprints add),
    then each endpoint's key coordinate (-1 for none) and footprint."""
    arm_a = int(0)
    if body_a >= 0 and art_a >= 0:
        if data.response_dofs[art_a] > 0 and data.response_dofs[art_a] != slots.star:
            arm_a = 1
    coord_a, z_a, walked_a = slot_seed(
        walk, support, slots, data, v_hat, sides, group, body_a, art_a, 1.0, row, direction, point, 0, slot
    )
    coord_b, z_b, walked_b = slot_seed(
        walk, support, slots, data, v_hat, sides, group, body_b, art_b, -1.0, row, direction, point, arm_a, slot
    )
    walked = walked_a + walked_b
    if arm_a != 0 and walked_b[1] > 0.0:
        walked = wp.vec2(walked[0], -1.0)
    return walked, coord_a, z_a, coord_b, z_b


@wp.func
def clear_dense_part(slots: SlotRows, group: int, slot: int):
    """Zero row ``slot``'s dense part before its seeds walk into it."""
    for c in range(slots.Z.shape[2]):
        slots.Z[group, slot, c] = 0.0


@wp.func
def slot_header(
    rows: DenseRows,
    support: RowSupport,
    slots: SlotRows,
    walk: WalkTrees,
    sides: wp.vec2i,
    group: int,
    line: int,
    slot: int,
    walked: wp.vec2,
    coord_a: int,
    z_a: float,
    coord_b: int,
    z_b: float,
    dt: float,
    contact_speculative_scale: float,
    contact_w: float,
    restitution_velocity_threshold: float,
    weighted: int,
    material: int,
):
    """Finish row ``slot`` once its seeds are walked (``walked``: its incident velocity and its arm footprints' sum of
    squares, negative where they must be re-read): its two slots, merged into the first on one coordinate; the arm's
    slips' release (:func:`release_side`); its diagonal ``(|z|^2 + sum_s z_s^2) + cfm``, incident velocity and rhs by the
    row law (:func:`.kernels.row_bias`)."""
    if coord_a >= 0 and coord_a == coord_b:
        z_a += z_b
        coord_b = -1
        z_b = 0.0
    slots.dof[line, slot, 0] = coord_a
    slots.dof[line, slot, 1] = coord_b
    slots.z[line, slot, 0] = z_a
    slots.z[line, slot, 1] = z_b
    norm = walked[1]
    if slips_on(walk, sides[0]) != 0:
        release_side(walk, support, sides, group, sides[0], slots.Z, slot)
        norm = -1.0
    if sides[1] != sides[0] and slips_on(walk, sides[1]) != 0:
        release_side(walk, support, sides, group, sides[1], slots.Z, slot)
        norm = -1.0
    if norm < 0.0:
        norm = float(0.0)
        for c in range(slots.Z.shape[2]):
            z = slots.Z[group, slot, c]
            norm += z * z
    sparse = float(0.0)
    if coord_a >= 0:
        sparse += z_a * z_a
    if coord_b >= 0:
        sparse += z_b * z_b
    rows.diag[line, slot] = (norm + sparse) + rows.row_cfm[line, slot]
    slots.incident[line, slot] = walked[0]
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


@wp.kernel(enable_backward=False)
def write_slot_rows(
    rows: DenseRows,
    support: RowSupport,
    slots: SlotRows,
    counts: wp.array[int],
    bounds: wp.array2d[int],
    prefix: int,
    lanes: int,
    data: ContactInput,
    law: PatchRowLaw,
    walk: WalkTrees,
    v_hat: wp.array[float],
    dt: float,
    contact_speculative_scale: float,
    contact_w: float,
    restitution_velocity_threshold: float,
    weighted: int,
    material: int,
):
    """The sparse diagonal's one writer: a world's ``lanes`` threads stride over its rows (``dim = (worlds, lanes)``),
    every row (``counts``, the allocation's watermark) or its limit prefix alone (``prefix``: ``bounds[:, 1]``, where the
    compact boundary writes the contact rows in its producer). Each row's law from its unit source: a contact row's two
    endpoints (:func:`contact_seeds`), a limit's covector on the arm (:func:`walk_covector`); then its header
    (:func:`slot_header`)."""
    world, lane = wp.tid()
    end = counts[world]
    if prefix != 0:
        end = bounds[world, 1]
    end = wp.min(end, support.units.shape[1])
    line = rows.first[world]
    group = slots.dense_group[world]
    for slot in range(lane, end, lanes):
        source = support.units[line, slot]
        kind = source >> UNIT_SHIFT
        index = source & ((1 << UNIT_SHIFT) - 1)
        sides = rows.sides[line, slot]
        clear_dense_part(slots, group, slot)
        walked = wp.vec2(0.0)
        coord_a = int(-1)
        z_a = float(0.0)
        coord_b = int(-1)
        z_b = float(0.0)
        if kind == UNIT_CONTACT:
            c = index // 4
            r = index - 4 * c
            # The row's records by the contact law (the producer's, folded into the writer), then its seeds.
            body_a, body_b, direction, row_point = write_contact_row(data, law, c, r, False)
            sides = rows.sides[line, slot]
            walked, coord_a, z_a, coord_b, z_b = contact_seeds(
                walk,
                support,
                slots,
                data,
                v_hat,
                sides,
                group,
                slot,
                r,
                direction,
                row_point,
                body_a,
                data.art0[c],
                body_b,
                data.art1[c],
            )
        elif kind == UNIT_LIMIT:
            dof = index // 2
            value = 1.0 - 2.0 * float(index - 2 * dof)
            walked = walk_covector(walk, support, sides, group, sides[0], dof, value, v_hat, 0, slots.Z, slot)
        slot_header(
            rows,
            support,
            slots,
            walk,
            sides,
            group,
            line,
            slot,
            walked,
            coord_a,
            z_a,
            coord_b,
            z_b,
            dt,
            contact_speculative_scale,
            contact_w,
            restitution_velocity_threshold,
            weighted,
            material,
        )

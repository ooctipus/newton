# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The one writer of the rows: every row a seed and a law, whitened by the walk, its placement a sink.

One unit walker writes every placement's rows (:data:`_SUPPORT`, the support writer's text). A unit is a contact's
rows walked together from its first (1, 3, or 4 under contact torsion), or a joint or link row alone
(:func:`unit_reader`); its seeds, a wrench at a body's frame or a covector at a coordinate (``WALK_WRENCH``,
:func:`joint_seeds`), walk up their root paths (``WALK_ROW``) into the placement's sink:

- **the support placement** (case K1, FL3.4): each footprint at its rank in the row's support (factor coordinates)
  and the unit's Gram for its block record, through the class's path table in constant memory; the step's units are
  listed first (:func:`support_units_kernel`, :class:`UnitList`) and the walkers stride over the list
  (:func:`support_rows_kernel`), the work sized by the step's units, not by the rows' capacity;
- **the general placement** (case K6, FL3.3): each side's entries in the packed rows, through each tree's path table
  (:func:`packed_walker`), world-major with the bilateral blocks' closure (:func:`walked_rows_kernel`);
- **the sparse diagonal** (case K2, FL3.5): the arm's side into the row's dense part, a key of the star one step to
  the row's slot (:func:`write_slot_rows`, :func:`slot_header`; the compact boundary's producer,
  :func:`.compact_contact.contacts_kernel`);
- **the free bodies' rows** (``mf_Z``, each side's six of a row's twelve: a packed row by the free bodies' own row): a
  contact row (:func:`write_free_body_rows`), or a speed row's axis, a covector, for its diagonal
  (:func:`populate_rigid_velocity_limit_rows`), through the packed walker (:func:`packed_walker`).

Each row's header is its diagonal ``|z|^2 + cfm`` (the walk's sums of squares, re-read where two seeds share a side or
a side slips), its incident velocity ``J v_hat`` (the step's ``J`` entries each walk passes) and its rhs by the row law
(:func:`.kernels.row_bias`). A wrench enters as the step's ``J`` (the step's columns the finish published, shifted to
the walk's frames, :attr:`.tree_factor.WalkTrees.now`), so a step between refreshes whitens the step's ``J`` by the
held factor. A slipping drive's ``K`` leaves the rows by its release along its path (``R``, the predictor's border,
:func:`.tree_kinetic_state.border_solve`), which keeps every row inside its support.

No ``J`` is formed. The rows and the sweeps' leave share the walk's one frame (FL3_design §1.2); the sweeps' change
goes to the step's one leave (:func:`.tree_factor.walk_leave_kernel`).
"""

from __future__ import annotations

from functools import cache

import warp as wp

from . import field_group
from .contact_input import ContactInput, contact_frame
from .contact_laws import CONTACT_SPIN_ROW, contact_row_entry, contact_seed, spin_seed
from .dense_rows import (
    UNIT_CONTACT,
    UNIT_LIMIT,
    UNIT_LINK,
    UNIT_MIMIC,
    UNIT_SHIFT,
    UNIT_VELOCITY,
    DenseRows,
    RowSupport,
    unit_source,
)
from .device_preludes import DevicePrelude, dual_native, lane_group, substitute
from .field_group import Field
from .friction_patches import PatchRowLaw
from .joint_table import FreeBodies
from .kernel_template import python_kernel
from .kernels import PGS_CONSTRAINT_TYPE_SPEED, RIGID_SPEED_ROWS, row_bias
from .projection_laws import friction_pair_metric
from .response_builds_rows import close_block, close_row
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
    walk_paths_prelude,
    walk_prelude,
    walk_row_prelude,
)
from .tree_scan import WALK_BOTH, held_fits, record_half

__all__ = [
    "SUPPORT_UNIT_LANES",
    "JointSeeds",
    "RowSink",
    "RowUnit",
    "SlotRows",
    "UnitList",
    "bind_slot_rows",
    "free_body_sink",
    "packed_sink",
    "packed_walker",
    "populate_rigid_velocity_limit_rows",
    "row_sink",
    "support_rows_kernel",
    "support_threads",
    "support_units_kernel",
    "unit_list",
    "unit_reader",
    "walk_slot_row",
    "walked_rows_kernel",
    "write_free_body_rows",
    "write_slot_rows",
    "writer_kernel",
    "writer_lanes",
]


@wp.func
def body_wrench(walk: WalkTrees, body: int, direction: wp.vec3, point: wp.vec3, torque: bool) -> wp.spatial_vector:
    """The seed of a row acting on ``body`` along ``direction`` at ``point`` (a pure ``torque`` about it), by the
    contact law (:func:`.contact_laws.contact_seed`, :func:`.contact_laws.spin_seed`): a wrench at the body's frame, the
    factor's (:attr:`.tree_factor.WalkTrees.link`), where the walk shifts the step's columns the finish published to
    take the step's ``J``. The lever arm is the world point less the frame point, which is the articulation's (relative to
    its origin, :attr:`.tree_factor.WalkTrees.origin`); the law takes it from wherever it is formed."""
    if torque:
        return spin_seed(direction)
    j = walk.body_joint[body]
    frame = wp.vec3(walk.link[j, WALK_FRAME], walk.link[j, WALK_FRAME + 1], walk.link[j, WALK_FRAME + 2])
    return contact_seed(direction, (point - walk.origin[walk.joint_art[j]]) - frame)


@wp.struct
class JointSeeds:
    """What the rows' seeds read besides the contact input (:func:`joint_seeds`, a link row's wrenches): each body's
    articulation, each mimic's two coordinates and the second's coefficient, each link's primary and shadow bodies and
    the bodies' poses and centres of mass, a link row's points. A placement without links or mimics binds it empty."""

    body_to_articulation: wp.array[int]
    mimic_dof0: wp.array[int]
    mimic_dof1: wp.array[int]
    mimic_coef1: wp.array[float]
    link_primary: wp.array[int]
    link_shadow: wp.array[int]
    body_q: wp.array[wp.transform]
    body_com: wp.array[wp.vec3]


@wp.func
def joint_seeds(seeds: JointSeeds, kind: int, index: int):
    """A joint row's covector seeds by its law, from its unit source (``kind``, ``index``): a limit's and a velocity
    limit's ``+-e_dof`` at ``dof0`` (side 0 the lower, ``+``; side 1 the upper, ``-``), a mimic's ``e_f - c e_l`` with
    its second covector at ``dof1`` (-1 for none). Every writer's dispatch of a joint row."""
    dof0 = index // 2
    value0 = 1.0 - 2.0 * float(index - 2 * dof0)
    dof1 = int(-1)
    value1 = float(0.0)
    if kind == UNIT_MIMIC:
        dof0 = seeds.mimic_dof0[index]
        value0 = 1.0
        dof1 = seeds.mimic_dof1[index]
        value1 = -seeds.mimic_coef1[index]
    return dof0, value0, dof1, value1


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


WRITER_THREADS = 128
"""Threads of a block of the one writer: whole world groups of up to 32 lanes."""

WRITER_BLOCKS = 6, 5
"""The one writer's blocks a multiprocessor (its launch bound), without and with bilateral blocks: the register line
its walk fits, for a latency-bound walk whose speed follows its resident warps. Without blocks (no link rows), six
(80 registers, 24 warps); with blocks, whose link rows hold a second point across the walk and whose closure follows
it, five (96 registers: 97 unbounded, and six spilled 88 B a thread, sm_120)."""


@cache
def unit_reader(links: bool, unit_rows: int = 1) -> wp.Function:
    """Every placement's unit reader (:func:`row_unit`). A contact's unit is its rows from its first, at most
    ``unit_rows``, from the contact's frame: the support placement's, whose unit list holds a contact's rows as one unit
    and whose first launch wrote their records (:func:`support_units_kernel`). With ``unit_rows`` 1 it is the row
    alone, its records written here by the contact law: the world-major writers', a lane a row (the producer's, folded
    into the writer). ``links``: with the link rows' seeds, where the writer closes bilateral blocks (the general
    placement's block variants; the sparse diagonal and the support placement admit none)."""

    def row_unit(
        rows: DenseRows, data: ContactInput, law: PatchRowLaw, seeds: JointSeeds, line: int, slot: int, source: int
    ):
        """The unit at row ``slot`` by its law (its unit source ``source``): a contact's rows, its bodies, frame rows
        and points (:func:`.contact_input.contact_frame`; a row alone, its direction and point,
        :func:`.row_build.write_contact_row`; the spin row a torque about the normal); a link's row, the unit wrench at
        its two pieces' centres of mass (the shadow's negated; a pure torque for an angular row); a joint limit's, a
        velocity limit's or a mimic's covectors (:func:`joint_seeds`)."""
        kind = source >> UNIT_SHIFT
        index = source & ((1 << UNIT_SHIFT) - 1)
        unit = RowUnit()
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
            unit.row = index - 4 * c
            if wp.static(unit_rows > 1):
                unit.rows = wp.min(data.slots_needed[c], wp.static(unit_rows))
                body_a, body_b, normal, point, _phi, tangent0, tangent1, friction_point = contact_frame(data, law, c)
                unit.frame = wp.matrix_from_rows(normal, tangent0, tangent1)
                unit.friction_point = friction_point
            else:
                body_a, body_b, direction, point = write_contact_row(data, law, c, unit.row, False)
                unit.frame = wp.matrix_from_rows(direction, direction, direction)
                unit.friction_point = point
            unit.art_a = data.art0[c]
            unit.body_a = body_a
            unit.art_b = data.art1[c]
            unit.body_b = body_b
            unit.point = point
            unit.point_b = point
            if unit.row == CONTACT_SPIN_ROW:
                unit.torque = 1
        elif wp.static(links) and kind == UNIT_LINK:
            k = index // LINK_ROWS
            row = index - LINK_ROWS * k
            # The row's axis by selects: written through a runtime index it would sit in local memory.
            axis = row % 3
            direction = wp.vec3(
                wp.where(axis == 0, 1.0, 0.0), wp.where(axis == 1, 1.0, 0.0), wp.where(axis == 2, 1.0, 0.0)
            )
            primary = seeds.link_primary[k]
            shadow = seeds.link_shadow[k]
            unit.art_a = seeds.body_to_articulation[primary]
            unit.body_a = primary
            unit.art_b = seeds.body_to_articulation[shadow]
            unit.body_b = shadow
            unit.frame = wp.matrix_from_rows(direction, direction, direction)
            unit.point = wp.transform_point(seeds.body_q[primary], seeds.body_com[primary])
            unit.point_b = wp.transform_point(seeds.body_q[shadow], seeds.body_com[shadow])
            unit.friction_point = unit.point
            if row >= 3:
                unit.torque = 1
        else:
            unit.art_a = rows.sides[line, slot][0]
            dof0, value0, dof1, value1 = joint_seeds(seeds, kind, index)
            unit.dof0 = dof0
            unit.value0 = value0
            unit.dof1 = dof1
            unit.value1 = value1
        return unit

    name = "row_unit" + ("_links" if links else "") + (f"_{unit_rows}" if unit_rows > 1 else "")
    row_unit.__name__ = row_unit.__qualname__ = name
    return wp.func(row_unit)


@cache
def walked_rows_kernel(blocks: int, mimic_blocks: bool, side: bool = True) -> wp.Kernel:
    """The one writer of the general placement, world-major: a world's group of ``lanes`` lanes (:func:`writer_lanes`)
    writes its rows in the block primitive's phases (FL3_design §1.4, W9), the group synchronized between them:

    1. *rows*: a lane a row, every row of the world but a world an in-kernel owner holds (``owned``: the local owners'
       packets walk their own seeds): a contact row's records by the contact law, then the row as its own unit
       (:func:`unit_reader`), walked by the one unit walker with the packed rows as its sink
       (:func:`packed_walker`; the spin row a torque), each side's slips released by the
       predictor's border's ``R`` (:func:`.tree_kinetic_state.border_solve`), then its rhs by the row law
       (:func:`.kernels.row_bias`). A row a lane, not a contact a lane: the world's lanes stay busy (the support
       placement's unit list spreads a contact's unit over the GPU instead, :func:`support_rows_kernel`);
    2. *closure* (``blocks``, the largest bilateral block's rows; 0 for none): the lanes close each of the world's
       blocks (:func:`.response_builds_rows.close_block`; its rows are the links', and the mimics' with
       ``mimic_blocks``), then correct every other row on the blocks its sides touch
       (:func:`.response_builds_rows.close_row`).

    The closure's seed stays a launch after the warm start's gather (:meth:`.tree_factor.TreeFactor.seed_blocks`).
    One launch with group barriers, not a second launch for the block phase: FL33D's measurement (``fl33d/``).
    ``side``: whether the model's records can have a side (:attr:`.tree_factor.TreeFactor.walk_side`)."""
    has_blocks = blocks > 0
    capacity = max(blocks, 1)
    mimics = int(mimic_blocks)
    # A link row is a block row (block_row; links plan blocks, TreeFactor.plan_blocks): the writer reads link rows
    # where it closes blocks. Elsewhere their seeds would hold a second point across every walk.
    read_unit = unit_reader(has_blocks)
    walk = packed_walker(1, False, side)

    def kernel(
        rows: DenseRows,
        support: RowSupport,
        counts: wp.array[int],
        owner: wp.array[int],
        owned: int,
        lanes: int,
        data: ContactInput,
        law: PatchRowLaw,
        sink: RowSink,
        preelim: Preelim,
        mimic_slot: wp.array[int],
        link_slot: wp.array[int],
        seeds: JointSeeds,
        dt: float,
        contact_speculative_scale: float,
        contact_w: float,
        restitution_velocity_threshold: float,
        weighted: int,
        material: int,
    ):
        world, lane = wp.tid()
        count = wp.min(counts[world], sink.entries.shape[1])
        line = rows.first[world]
        # A world an in-kernel owner holds (``owned`` and ``owner`` nonzero: the local owners' packets, whose kernels
        # walk their seeds) is not the writer's: its whole group returns before any barrier.
        if owned != 0 and owner[world] != 0:
            return
        for slot in range(lane, count, lanes):
            unit = read_unit(rows, data, law, seeds, line, slot, support.units[line, slot])
            if unit.rows > 0:
                walk(sink, unit)
                row_bias(
                    rows, line, slot, sink.row_incident[line, slot], dt, contact_speculative_scale, contact_w,
                    restitution_velocity_threshold, weighted, material,
                )  # fmt: skip
        if wp.static(has_blocks):
            world_sync(lanes)
            for pe in range(preelim.world_start[world] + lane, preelim.world_start[world + 1], lanes):
                close_block(
                    pe, wp.static(capacity), line, preelim, mimic_slot, link_slot, rows.cfm, sink.entries,
                    rows.rhs, sink.row_incident, rows.diag,
                )  # fmt: skip
            world_sync(lanes)
            for slot in range(lane, count, lanes):
                if block_row(support.units[line, slot], wp.static(mimics)) == 0:
                    close_row(
                        line, slot, rows.sides[line, slot], wp.static(capacity), preelim, rows.cfm, sink.entries,
                        rows.rhs, rows.diag,
                    )  # fmt: skip

    name = "write_walked_rows" + (f"_blocks{blocks}" if has_blocks else "")
    name += "_mimics" if mimic_blocks else ""
    return python_kernel(kernel, name, launch_bounds=(WRITER_THREADS, WRITER_BLOCKS[int(has_blocks)]))


def writer_kernel(solver) -> wp.Kernel:
    """The one writer's variant for the solver's class (:func:`walked_rows_kernel`), from its topology: the block phase
    where bilateral blocks exist (mimics among them under ``enable_bilateral_preelimination``), and the records' side
    where a walked joint can have two halves (:attr:`.tree_factor.TreeFactor.walk_side`)."""
    factor = solver.factor
    blocks = factor.block_capacity if factor.eliminates else 0
    mimics = bool(blocks and solver.options.enable_bilateral_preelimination)
    return walked_rows_kernel(blocks, mimics, factor.walk_side)


# ---------------------------------------------------------------------------------------------------------------------
# The support placement (case K1, FL3.4): the support sink, a lane per unit, the release and the leave
# ---------------------------------------------------------------------------------------------------------------------


_SUPPORT_TABLE = """    static WALK_CONSTANT int SUPPORT_NODE[$n] = {$node};"""

_SUPPORT = """    // One unit's rows into the support sink (walk_rows.support_rows_kernel): its seeds walked through the class's
    // path table (WALK_ROW), each footprint stored at its rank in the row's support (SUPPORT_NODE: a
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
    // The writer's own accesses, typed (no generic LD.E / ST.E): its inputs read-only (RD, the non-coherent path;
    // nothing here writes them), its outputs stored (ST), and the footprints it adds into read back (RW).
#if defined(__CUDA_ARCH__)
    const auto RD = [](const auto* p) { return $rd; };
    const auto ST = [](auto* p, const auto v) {
        __builtin_assume(__isGlobal(p));
        *p = v;
    };
    const auto RW = [](const float* p) -> float {
        __builtin_assume(__isGlobal(p));
        return *p;
    };
#else
    const auto RD = [](const auto* p) { return *p; };
    const auto ST = [](auto* p, const auto v) { *p = v; };
    const auto RW = [](const float* p) -> float { return *p; };
#endif
    const auto& walk = sink.walk;
    const auto& release = sink.release;
    const auto& support = sink.support;
    const int rows = unit.rows, art_a = unit.art_a, art_b = unit.art_b, body_a = unit.body_a, body_b = unit.body_b;
    const int dof0 = unit.dof0, dof1 = unit.dof1;
    const int S = sink.entries.shape[2];
    const int row0 = unit.line * sink.entries.shape[1] + unit.slot;
    // The row's records in the row store, whose line holds the free bodies' rows after the dense ones.
    const int rec0 = unit.line * sink.diag.shape[1] + unit.slot;
    float* const Z = sink.entries.data + (size_t)$zrow * S;
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
$start    for (int side = 0; side < 2; ++side) {
        const int art = side == 0 ? art_a : art_b;
        if (art < 0 || (side == 1 && art == art_a) || RD(sink.response_dofs.data + art) <= 0) continue;
        const int j0 = RD(walk.joint_start.data + art), d0 = RD(walk.dof_start.data + art);
$side
        const float* data = walk.data.data + (size_t)d0 * $coordinate;
        const float* link = walk.link.data + (size_t)j0 * $link;
        const float* now = reinterpret_cast<const float*>(walk.now.data + d0);
        const float* now_link = walk.now_link.data + (size_t)j0 * $link;
        const float* vh = sink.v_hat.data + d0;
        const auto rank = [&](int c) -> int { return $rank; };
        // A coordinate's footprints leave together after its last row (the walk gives each coordinate's rows in
        // turn): no store between a coordinate's rows, so the walk's loads of that coordinate serve all its rows.
        float footprint[$rows];
        const auto store = [&](int r, int c, float z, float) {
            if (r < rows) {
                footprint[r] = z;
                if (!add) {
                    norm[r] += z * z;
                    zc[r] = z;
                }
            }
            if (r != $last) return;
            float* entry = Z + rank(c);
            for (int k = 0; k < rows; ++k) {
                if (add)
                    ST(entry + k * S, RW(entry + k * S) + footprint[k]);
                else
                    ST(entry + k * S, footprint[k]);
            }
            if (!add && rows == $rows) gram();
        };
        if (covector) {
            for (int k = 0; k < seeds; ++k) {
                const int c = (k == 0 ? dof0 : dof1) - d0;
                float x[$rows];
                for (int r = 0; r < $rows; ++r) x[r] = 0.0f;
                x[0] = k == 0 ? unit.value0 : unit.value1;
                // A covector enters at its coordinate: its body's coordinates after it (below it in factor order) are
                // the row's zeros.
$below                WALK_ROW(data, link, nullptr, nullptr, nullptr, RD(walk.dof_body.data + d0 + c), c, x, store);
                incident[0] += x[0] * RD(vh + c);
            }
        } else {
            float seed[6 * $rows];
            for (int k = 0; k < 2; ++k) {
                const int body = k == 0 ? body_a : body_b;
                if (body < 0 || (k == 0 ? art_a : art_b) != art) continue;
                const float sign = k == 0 ? 1.0f : -1.0f;
                const int j = RD(walk.body_joint.data + body);
                const int local = j - j0;
                // The seeds' wrenches at the body's frame (the factor's), where the walk's shifted columns are J's: the
                // frame point is the articulation's, so the world point meets it through the articulation's origin.
                const float* o = walk.link.data + (size_t)j * $link + $frame;
                const float* origin = reinterpret_cast<const float*>(walk.origin.data + art);
                const float frame_o[3] = {RD(o), RD(o + 1), RD(o + 2)};
                const float origin_o[3] = {RD(origin), RD(origin + 1), RD(origin + 2)};
                for (int r = 0; r < $rows; ++r) {
                    float* q = seed + 6 * r;
                    if (r >= rows) {
                        for (int i = 0; i < 6; ++i) q[i] = 0.0f;
                        continue;
                    }
$wrench
                }
                // The step's J entry of each footprint against v_hat: the rows' incident velocity.
                const auto sink = [&](int r, int c, float z, float j) {
                    if (r < rows) incident[r] += j * RD(vh + c);
                    store(r, c, z, j);
                };
                float shift[3];
                WALK_SHIFT(walk, art, shift);
                WALK_ROW(data, link, now, now_link, shift, local, -1, seed, sink);
            }
        }
        // A slipping tree's release (the predictor's border), root-most slip first, on the side's entries along each
        // slip's path (WALK_RELEASE_ON): entries off the row's support stay zero, so the side keeps its support.
        if (RD(release.count.data + art) > 0) {
            reread = true;
$release
        }
    }
$close    if (reread) {
        for (int r = 0; r < $rows; ++r) norm[r] = zc[r] = 0.0f;
        for (int k = 0; k < 6; ++k) cross[k] = 0.0f;
        for (int e = 0; e < $entries_end; ++e) {
            for (int r = 0; r < rows; ++r) {
                zc[r] = RW(Z + r * S + e);
                norm[r] += zc[r] * zc[r];
            }
            if (rows == $rows) gram();
        }
    }
    for (int r = 0; r < rows; ++r) {
        ST(sink.diag.data + rec0 + r, $diag);
        ST(sink.row_incident.data + rec0 + r, incident[r]);
    }
$record"""
"""The support sink's unit walker, ``$rows`` rows a lane (:func:`support_rows_kernel`)."""

_SUPPORT_CLOSE = """    // The pad past the support; the rows' coordinates (explicit) or their support class (the table).
    for (int r = 0; r < rows; ++r)
        for (int e = count; e < S; ++e) ST(Z + r * S + e, 0.0f);
    if (support.enabled == 0) {
        int* const D = sink.row_dof.data + (size_t)row0 * S;
        for (int side = 0; side < 2; ++side) {
            if (masks[side] == 0ull) continue;
            int e = bases[side];
            const int offset = RD(sink.world_offset.data + (side == 0 ? art_a : art_b));
            for (unsigned long long bits = masks[side]; bits != 0ull; bits &= bits - 1ull, ++e)
                for (int r = 0; r < rows; ++r) ST(D + r * S + e, offset + LOW(bits));
        }
        for (int r = 0; r < rows; ++r)
            for (int e = count; e < S; ++e) ST(D + r * S + e, -1);
    } else {
        int cls = 0;
        if (covector) {
            cls = RD(support.limit_class.data + (dof0 - RD(walk.dof_start.data + art_a)));
        } else {
            const auto side_of = [&](int body, int art) {
                return body >= 0 && art >= 0 && RD(sink.response_dofs.data + art) > 0 && RD(sink.body_mask.data + body) != 0ull;
            };
            const bool a = side_of(body_a, art_a), b = side_of(body_b, art_b);
            const int class_a = a ? RD(support.body_class.data + body_a) : 0;
            const int class_b = b ? RD(support.body_class.data + body_b) : 0;
            cls = RD(support.pair_class.data + (class_a * support.pair_class.shape[1] + class_b));
        }
        for (int r = 0; r < rows; ++r) ST(support.row_class.data + row0 + r, cls);
    }
"""

_SUPPORT_RECORD = """    // A full unit's block record from its Gram (sparse_contact.CONTACT_RECORD_FLOATS): its couplings W_nt1, W_nt2,
    // W_t1t2 and 1 / W_nn, then the spin row's couplings W_ns, W_t1s, W_t2s (contact torsion) or, written by the
    // kernel from these, the tangent pair's metric (projection_laws.friction_pair_metric).
    if (rows == $rows) {
        float* record = sink.records.data + ((size_t)unit.line * sink.records.shape[1] + unit.slot / 3)
                        * sink.records.shape[2];
        ST(record + 0, cross[0]);
        ST(record + 1, cross[1]);
        ST(record + 2, cross[2]);
        ST(record + 3, 1.0f / (norm[0] + sink.cfm));
$tail
    }
"""
_SPIN_TAIL = """        ST(record + 4, cross[3]);
        ST(record + 5, cross[4]);
        ST(record + 6, cross[5]);"""


@cache
def _support_table(node: tuple[int, ...]) -> DevicePrelude:
    """The class's factor coordinate of each tree coordinate, in constant memory (``WALK_CONSTANT``, after the walk's
    prelude): the support mask's bit."""
    text = substitute(_SUPPORT_TABLE, n=len(node), node=", ".join(str(int(k)) for k in node))
    return DevicePrelude(f"support_table_{len(node)}", text, cpu=text)


_RELEASE_HELD = """            // The side's entries held in the thread's own array while every slip's steps run over them: one load and
            // one store an entry, not a global read-modify-write a step.
            float held[$rows * $coordinates];
            const int span = count - base;
            for (int r = 0; r < rows; ++r)
                for (int e = 0; e < span; ++e) held[r * $coordinates + e] = RW(Z + r * S + base + e);
            WALK_RELEASE_ON(release, art, held, rows, $coordinates, [&](const int c) -> int {
                return ((mask >> SUPPORT_NODE[c]) & 1ull) != 0ull ? rank(c) - base : -1;
            });
            for (int r = 0; r < rows; ++r)
                for (int e = 0; e < span; ++e) ST(Z + r * S + base + e, held[r * $coordinates + e]);"""
_RELEASE_IN_PLACE = """            WALK_RELEASE_ON(release, art, Z, rows, S, [&](const int c) -> int {
                return ((mask >> SUPPORT_NODE[c]) & 1ull) != 0ull ? rank(c) : -1;
            });"""

SUPPORT_BLOCK, SUPPORT_BLOCKS = 256, 2
"""The support writer's block and its blocks a multiprocessor (its launch bound): 128 registers a thread."""

SUPPORT_REGISTERS = 65536 // (SUPPORT_BLOCK * SUPPORT_BLOCKS)
"""The support writer's register budget a thread, from its launch bound: what a unit's held side must fit
(:func:`.tree_scan.held_fits`, :data:`_RELEASE_HELD`).

The held side is a unit's at most four rows over its class's coordinates. Measured (``SP/core/FL/HANDOFF_ZK.md``
§17): franka-reach's 36 words held, its writer 16% faster (GB300); g1's 172 words held, 3% slower (GB300), so g1's
release runs in place."""


_SUPPORT_SIDE = """        unsigned long long mask = 0ull;
        int seeds = 0;
        if (covector) {
            mask = RD(sink.dof_mask.data + (dof0 - d0));
            seeds = 1;
            if (dof1 >= 0) {
                mask |= RD(sink.dof_mask.data + (dof1 - d0));
                seeds = 2;
            }
        } else {
            if (body_a >= 0 && art_a == art) {
                mask |= RD(sink.body_mask.data + body_a);
                ++seeds;
            }
            if (body_b >= 0 && art_b == art) {
                mask |= RD(sink.body_mask.data + body_b);
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
                for (int e = base; e < count; ++e) ST(Z + r * S + e, 0.0f);
        }"""

_HEADER_DIAG = "norm[r] + sink.cfm"

_SUPPORT_BELOW = """                if (!add)
                    for (int e = base; e < rank(c); ++e) ST(Z + e, 0.0f);
"""

_SUPPORT_WRENCH = """                    const auto& at = r == 0 ? unit.point : unit.friction_point;
                    WALK_WRENCH(q, sign, unit.frame.data[r == 1 || r == 2 ? r : 0], at.c, origin_o, frame_o,
                                $spin && r == 3);"""


def _support_snippet(parent: tuple, first: tuple, node: tuple, rows: int, records: bool, spin: bool, half: int) -> str:
    record = ""
    if records:
        record = substitute(_SUPPORT_RECORD, rows=rows, tail=_SPIN_TAIL if spin else "")
    held = held_fits(4 * first[-1], SUPPORT_REGISTERS)
    release = substitute(_RELEASE_HELD, rows=rows, coordinates=first[-1]) if held else _RELEASE_IN_PLACE
    body = substitute(
        _SUPPORT,
        rows=rows,
        release=release,
        last=rows - 1,
        coordinate=WALK_COORDINATE,
        link=WALK_LINK,
        frame=WALK_FRAME,
        record=record,
        start="",
        side=_SUPPORT_SIDE,
        rank="base + POPC(mask & ((1ull << SUPPORT_NODE[c]) - 1ull))",
        below=_SUPPORT_BELOW,
        wrench=substitute(_SUPPORT_WRENCH, spin=int(spin)),
        close=_SUPPORT_CLOSE,
        entries_end="count",
        zrow="row0",
        diag=_HEADER_DIAG,
        rd="__ldg(p)",
    )
    walk = walk_prelude(parent, first, rows, half)
    return dual_native(body, preludes=(walk, walk_row_prelude(rows), _support_table(node)))


@wp.struct
class RowSink:
    """What a placement's writer reads and writes besides a unit's seeds, every placement's (:data:`_SUPPORT`): the walk
    and its release, each articulation's response coordinates, the predictor ``v_hat``, and the rows' ``cfm``, entries,
    incident velocities and diagonals; the support placement's (case K1, :func:`support_rows_kernel`) support table,
    each body's and coordinate's support mask (factor coordinates), each articulation's first world coordinate, and the
    rows' coordinates and block records; the packed rows' layout, where each side's entries sit in a packed row
    (:class:`.dense_rows.RowSupport`, :func:`packed_walker`), the star's coordinates (the sparse diagonal's), and the
    free bodies' first row in the store's line (``base``: their entries index by their own row). A placement binds what
    its writer reads."""

    walk: WalkTrees
    release: WalkRelease
    response_dofs: wp.array[int]
    v_hat: wp.array[float]
    cfm: float
    entries: wp.array3d[float]
    row_incident: wp.array2d[float]
    diag: wp.array2d[float]
    support: _SparseRowSupport
    body_mask: wp.array[wp.uint64]
    dof_mask: wp.array[wp.uint64]
    world_offset: wp.array[int]
    row_dof: wp.array3d[int]
    records: wp.array3d[float]
    layout: RowSupport
    star: int
    base: int


@wp.struct
class RowUnit:
    """One unit's rows (``line``, ``slot``, ``rows``) and seeds, every placement's (:func:`unit_reader`): a contact's two
    bodies and their articulations with its frame's rows (the normal, then the tangents) at its ``point`` (body b's at
    ``point_b``: a link row's second piece) and ``friction_point``, a pure ``torque`` (a link's angular row), or a joint
    row's covectors ``value e_dof`` (``dof0 >= 0``; a mimic's second at ``dof1``) on articulation ``art_a``; its dense
    part's size group (``group``, the sparse diagonal's)."""

    line: int
    group: int
    slot: int
    rows: int
    row: int
    art_a: int
    body_a: int
    art_b: int
    body_b: int
    frame: wp.mat33
    point: wp.vec3
    point_b: wp.vec3
    friction_point: wp.vec3
    torque: int
    dof0: int
    value0: float
    dof1: int
    value1: float


def unit_walker(name: str, snippet: str) -> wp.Function:
    """The one unit walker's native function of name ``name`` over a placement's sink and a unit (:class:`RowSink`,
    :class:`RowUnit`): ``snippet`` is :data:`_SUPPORT` with the placement's slots filled."""

    def native(sink: RowSink, unit: RowUnit): ...

    native.__name__ = native.__qualname__ = name
    return wp.func_native(snippet)(native)


_PACKED_START = """    // The unit's packed rows cleared: each side's walk stores its path's entries, the rest of the row stays zero.
    for (int r = 0; r < rows; ++r)
        for (int e = 0; e < S; ++e) ST(Z + r * S + e, 0.0f);
    // The row's first side (its layout, dense_rows.row_sides), read after its records' writer wrote it.
    const int first = RD(sink.layout.sides.data + ((size_t)unit.line * sink.layout.sides.shape[1] + unit.slot)).c[0];
"""

_PACKED_SIDE = """$skip        WALK_TREE(reinterpret_cast<const int*>(walk.path.data + j0), RD(walk.joint_end.data + art) - j0,
                  RD(walk.dofs.data + art));
        int seeds = 0;
        if (covector) {
            seeds = dof1 >= 0 ? 2 : 1;
        } else {
            if (body_a >= 0 && art_a == art) ++seeds;
            if (body_b >= 0 && art_b == art) ++seeds;
        }
        if (seeds == 0) continue;
        // The side's first entry (dense_rows.jacobian_row): in a packed row the articulation's first world coordinate
        // (dense) or after the row's first side; in a size group's dense part (the star's arm) entry 0.
        const int base = $base;
        // Two seeds on one tree add where their paths meet, then the rows are re-read for their norms.
        const bool add = seeds > 1;
        if (add) reread = true;"""

_PACKED_BASE = """sink.layout.dense != 0 ? RD(sink.layout.offset.data + art) : (art == first ? 0 : RD(sink.layout.dofs.data + first))"""

_STAR_SKIP = """        // A key of the star is one step to the world, into the row's slot (star_step): the writer's, not the walk's.
        if (RD(sink.response_dofs.data + art) == sink.star) continue;
"""

_PACKED_WRENCH = """                    const auto& at = r == 0 ? (k == 0 ? unit.point : unit.point_b) : unit.friction_point;
                    WALK_WRENCH(q, sign, unit.frame.data[r == 1 || r == 2 ? r : 0], at.c, origin_o, frame_o,
                                ($spin && r == 3) || (r == 0 && unit.torque != 0));"""

_PACKED_RELEASE = (
    """            WALK_RELEASE_ON(release, art, Z, rows, S, [&](const int c) -> int { return rank(c); });"""
)


@cache
def packed_walker(rows: int, spin: bool, side: bool, star: bool = False, bodies: bool = False) -> wp.Function:
    """The general placement's unit walker (:data:`_SUPPORT`, the support writer's text with the packed rows as its
    sink): a unit's ``rows`` rows walked together through each tree's path table in global memory
    (:func:`.tree_factor.walk_paths_prelude`, a world holds any forest), each footprint stored at its coordinate's entry
    in the side's packed row, its diagonal ``|z|^2 + cfm`` and incident ``J v_hat`` from the same walk; ``spin``: the
    unit's fourth row is its spin row; ``side``: whether records can have a side. With ``star``, the sparse diagonal's:
    the arm's side into the row's dense part (its size group's row, ``unit.group``), a key's side left to the writer
    (:func:`star_step`), and the diagonal stored without its slots' squares and ``cfm``, which the writer adds. With
    ``bodies``, the free bodies': their packed rows by their own row, the row's less ``sink.base``
    (:func:`write_free_body_rows`)."""
    body = substitute(
        _SUPPORT,
        rows=rows,
        release=_PACKED_RELEASE,
        last=rows - 1,
        coordinate=WALK_COORDINATE,
        link=WALK_LINK,
        frame=WALK_FRAME,
        record="",
        start=_PACKED_START,
        side=substitute(
            _PACKED_SIDE,
            skip=_STAR_SKIP if star else "",
            base="0" if star else _PACKED_BASE,
        ),
        rank="base + c",
        below="",
        wrench=substitute(_PACKED_WRENCH, spin=int(spin)),
        close="",
        entries_end="S",
        zrow="(unit.group * sink.entries.shape[1] + unit.slot)" if star else "(row0 - sink.base)" if bodies else "row0",
        diag="norm[r]" if star else _HEADER_DIAG,
        # RD a plain load here, not the support writer's non-coherent __ldg: the K2 writer ran 1.4% faster on sm_120.
        rd="*p",
    )
    walk = walk_paths_prelude(rows, half=record_half(side))
    tag = ("_spin" if spin else "") + walk.name.removeprefix(f"walk_paths_{rows}") + ("_star" if star else "")
    tag += "_bodies" if bodies else ""
    snippet = dual_native(body, preludes=(walk, walk_row_prelude(rows)))
    return unit_walker(f"packed_walk_{rows}{tag}", snippet)


SUPPORT_UNIT_LANES = 16
"""A world's lane group in the support writer's first launch (:func:`support_units_kernel`): its rows' records and
units, a lane a row; the walkers' lanes (:func:`support_threads`) do not depend on it."""

UNIT_BUCKETS = 16
"""The unit list's buckets (:class:`UnitList`): a world's units go to bucket ``world % UNIT_BUCKETS``, so the lane
groups' reservations spread over that many counters of each kind."""


@wp.struct
class UnitList:
    """The step's units, one fixed slot each (:func:`support_units_kernel` appends, :func:`support_rows_kernel`
    walks). A unit is a row's seed and the rows it walks together, at least one row, so a world's units fit its rows'
    ``capacity``: the list holds every world's at that bound and cannot overflow. Bucket ``b`` holds its worlds' units
    in ``span`` slots, the wide units (two rows or more: a contact's) from the front and the one-row units from the
    back, each as ``line * capacity + slot`` of its first row; ``count[2 b]`` and ``count[2 b + 1]`` are its wide and
    one-row counts, zeroed before the step's append. The order within a bucket is the lane groups' arrival, on which
    no unit's rows depend."""

    entry: wp.array[int]
    count: wp.array[int]
    span: int
    capacity: int


def unit_list(worlds: int, capacity: int, device) -> UnitList:
    """An empty :class:`UnitList` for ``worlds`` worlds of ``capacity`` rows each."""
    units = UnitList()
    units.capacity = int(capacity)
    units.span = -(-int(worlds) // UNIT_BUCKETS) * int(capacity)
    units.entry = wp.empty(max(UNIT_BUCKETS * units.span, 1), dtype=int, device=device)
    units.count = wp.zeros(2 * UNIT_BUCKETS, dtype=int, device=device)
    return units


_UNIT_APPEND = """    // This round's units of the world's lane group into its bucket: one reservation a kind for the group.
    const unsigned wide_votes = BALLOT(wide != 0), one_votes = BALLOT(one != 0);
    const int wide_first = RESERVE(units.count.data + 2 * bucket, COUNT(wide_votes));
    const int one_first = RESERVE(units.count.data + 2 * bucket + 1, COUNT(one_votes));
    int* const region = units.entry.data + (size_t)bucket * units.span;
    if (wide != 0) region[wide_first + BELOW(wide_votes)] = entry;
    if (one != 0) region[units.span - 1 - (one_first + BELOW(one_votes))] = entry;"""


@wp.func_native(dual_native(_UNIT_APPEND, preludes=(lane_group("lanes"),)))
def unit_append(lanes: int, units: UnitList, bucket: int, wide: int, one: int, entry: int):
    """Append this lane's unit, ``wide`` (two rows or more) or ``one`` (one row), to bucket ``bucket`` of ``units``
    (:class:`UnitList`); every lane of the world's group of ``lanes`` calls it together."""


@wp.func
def unit_total(units: UnitList) -> int:
    """The list's units over its buckets."""
    total = int(0)
    for k in range(2 * UNIT_BUCKETS):
        total += units.count[k]
    return total


@wp.func
def unit_entry(units: UnitList, item: int) -> int:
    """The list's ``item``-th unit: its wide units first, bucket by bucket, then its one-row units."""
    wide = int(0)
    for b in range(UNIT_BUCKETS):
        wide += units.count[2 * b]
    kind = int(0)
    rank = item
    if item >= wide:
        kind = 1
        rank = item - wide
    b = int(0)
    n = units.count[kind]
    while rank >= n:
        rank -= n
        b += 1
        n = units.count[2 * b + kind]
    at = rank
    if kind == 1:
        at = units.span - 1 - rank
    return units.entry[b * units.span + at]


@cache
def support_units_kernel(unit_rows: int, lanes: int) -> wp.Kernel:
    """The support placement's units (case K1, the writer's first launch): ``dim = (worlds, lanes)``, a world's lane
    group of ``lanes`` striding over its rows, a lane a row: each contact row's records by the contact law
    (:func:`.row_build.write_contact_row`, its unit source written where its slot is reserved,
    :func:`.kernels.allocate_world_contact_slots`), and each unit's first row appended to the step's
    :class:`UnitList`: a contact's ``min(slots_needed, unit_rows)`` rows as one unit, a joint row (a limit, a velocity
    limit, a mimic) alone. The walkers (:func:`support_rows_kernel`) read the records after this launch."""

    def kernel(
        rows: DenseRows,
        sources: wp.array2d[int],
        counts: wp.array[int],
        data: ContactInput,
        law: PatchRowLaw,
        units: UnitList,
    ):
        world, lane = wp.tid()
        line = rows.first[world]
        count = wp.min(counts[world], units.capacity)
        bucket = world % UNIT_BUCKETS
        for base in range(0, count, wp.static(lanes)):
            slot = base + lane
            wide = int(0)
            one = int(0)
            if slot < count:
                source = sources[line, slot]
                kind = source >> UNIT_SHIFT
                index = source & ((1 << UNIT_SHIFT) - 1)
                if kind == UNIT_CONTACT:
                    c = index // 4
                    row = index - 4 * c
                    write_contact_row(data, law, c, row, False)
                    if row == 0:
                        if wp.min(data.slots_needed[c], wp.static(unit_rows)) > 1:
                            wide = 1
                        else:
                            one = 1
                elif kind == UNIT_LIMIT or kind == UNIT_VELOCITY or kind == UNIT_MIMIC:
                    one = 1
            unit_append(wp.static(lanes), units, bucket, wide, one, line * units.capacity + slot)

    return python_kernel(kernel, f"support_units_{unit_rows}_{lanes}")


@cache
def support_rows_kernel(
    parent: tuple[int, ...],
    first: tuple[int, ...],
    node: tuple[int, ...],
    *,
    records: bool,
    torsion: bool,
    half: int = WALK_BOTH,
) -> wp.Kernel:
    """The support placement's walkers (case K1, the writer's second launch): ``threads`` lanes, sized to the GPU,
    striding over the step's :class:`UnitList` (:func:`support_units_kernel`), its wide units first, so a warp runs one
    walker. A lane walks a unit: a contact's rows from its first, a joint row alone. The generator form (Gamma per body,
    ``z = Gamma w``) measured slower on g1 than this direct walk, so the writer walks each unit.

    The class's tree (``parent``, ``first``: :func:`.tree_factor.walk_prelude`'s table) and each coordinate's factor
    coordinate ``node`` (the support masks' bit) are baked into constant memory. A contact unit of three rows (four
    under contact ``torsion``) is walked together and its Gram writes its block record where ``records`` (the
    triplet sweeps'); a unit of one row (a normal alone, a limit, a velocity limit, a mimic) by the one-row walker. The
    contact rows' records are the first launch's; a joint row's records and unit source are its producer's (the limit,
    velocity-limit and mimic producers). The walker stores its footprints at their ranks in its support
    (``row_factor``, with the coordinates ``row_dof`` or the support class ``row_class``), its diagonal ``|z|^2 + cfm``,
    its incident ``J v_hat`` and its rhs by the row law (:func:`.kernels.row_bias`). ``half`` is the class's records'
    half (:func:`.tree_scan.classes_half`), compiled into its walks.
    """
    unit_rows = 4 if torsion else 3

    def walker(rows: int, with_records: bool):
        snippet = _support_snippet(parent, first, node, rows, with_records, torsion and rows == 4, half)
        return unit_walker(f"support_walk_{rows}", snippet)

    walk_one = walker(1, False)
    walk_unit = walker(unit_rows, records)
    read_unit = unit_reader(False, unit_rows)

    def kernel(
        rows: DenseRows,
        sources: wp.array2d[int],
        units: UnitList,
        threads: int,
        data: ContactInput,
        law: PatchRowLaw,
        sink: RowSink,
        seeds: JointSeeds,
        dt: float,
        contact_speculative_scale: float,
        contact_w: float,
        restitution_velocity_threshold: float,
        weighted: int,
        material: int,
    ):
        # The list's counts are read again at each unit (L1 hits): nothing but the item lives across a walk.
        item = wp.tid()
        while item < unit_total(units):
            entry = unit_entry(units, item)
            line = entry // units.capacity
            slot = entry - line * units.capacity
            unit = read_unit(rows, data, law, seeds, line, slot, sources[line, slot])
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
            item += threads

    name = f"support_rows_{len(node)}_{abs(hash((parent, first, node))) % (1 << 32):08x}"
    name += ("_records" if records else "") + ("_torsion" if torsion else "")
    # Two blocks of 256 a multiprocessor within the register file (128 registers): the four-row walker holds 166
    # unbounded, and the bound costs it 16 B of stack (sm_120, ptxas).
    return python_kernel(kernel, name, launch_bounds=(SUPPORT_BLOCK, SUPPORT_BLOCKS))


def support_threads(units: int, device) -> int:
    """The walkers' lanes (:func:`support_rows_kernel`): two waves of its two blocks of 256 a multiprocessor, at most
    ``units`` (the list's bound); one thread on the CPU."""
    device = wp.get_device(device)
    if not device.is_cuda:
        return 1
    return max(min(2 * SUPPORT_BLOCKS * SUPPORT_BLOCK * int(device.sm_count), int(units)), 1)


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
def slot_header(
    rows: DenseRows,
    slots: SlotRows,
    data: ContactInput,
    v_hat: wp.array[float],
    unit: RowUnit,
    dt: float,
    contact_speculative_scale: float,
    contact_w: float,
    restitution_velocity_threshold: float,
    weighted: int,
    material: int,
):
    """Finish the sparse diagonal's row ``unit.slot`` once the walk wrote its arm's footprints, norm and incident
    (:func:`packed_walker` with the star): each endpoint on a key of the star, one step to the world into the row's two
    slots (:func:`star_step`, merged into the first on one coordinate), its diagonal ``(|z|^2 + sum_s z_s^2) + cfm``, its
    incident velocity with the keys' and its rhs by the row law (:func:`.kernels.row_bias`)."""
    line = unit.line
    slot = unit.slot
    row = unit.row
    coord_a = int(-1)
    z_a = float(0.0)
    coord_b = int(-1)
    z_b = float(0.0)
    incident = slots.incident[line, slot]
    if unit.dof0 < 0:
        direction = unit.frame[0]
        point = unit.point
        if unit.body_a >= 0 and unit.art_a >= 0:
            if data.response_dofs[unit.art_a] == slots.star:
                coord_a, z_a, key = star_step(slots, data, v_hat, unit.art_a, unit.body_a, 1.0, row, direction, point)
                incident += key
        if unit.body_b >= 0 and unit.art_b >= 0:
            if data.response_dofs[unit.art_b] == slots.star:
                coord_b, z_b, key = star_step(slots, data, v_hat, unit.art_b, unit.body_b, -1.0, row, direction, point)
                incident += key
    if coord_a >= 0 and coord_a == coord_b:
        z_a += z_b
        coord_b = -1
        z_b = 0.0
    slots.dof[line, slot, 0] = coord_a
    slots.dof[line, slot, 1] = coord_b
    slots.z[line, slot, 0] = z_a
    slots.z[line, slot, 1] = z_b
    sparse = float(0.0)
    if coord_a >= 0:
        sparse += z_a * z_a
    if coord_b >= 0:
        sparse += z_b * z_b
    rows.diag[line, slot] = (rows.diag[line, slot] + sparse) + rows.cfm
    slots.incident[line, slot] = incident
    row_bias(
        rows, line, slot, incident, dt, contact_speculative_scale, contact_w, restitution_velocity_threshold, weighted,
        material,
    )  # fmt: skip


read_row_unit = unit_reader(False)
"""The unit reader of a row a unit, without link rows (:func:`unit_reader`): the sparse diagonal's and the free
bodies'."""

walk_slot_row = packed_walker(1, False, True, True)
"""The one unit walker with the star (:func:`packed_walker`): a row of the sparse diagonal."""


@wp.kernel(enable_backward=False)
def write_slot_rows(
    rows: DenseRows,
    support: RowSupport,
    slots: SlotRows,
    sink: RowSink,
    counts: wp.array[int],
    bounds: wp.array2d[int],
    prefix: int,
    lanes: int,
    data: ContactInput,
    law: PatchRowLaw,
    v_hat: wp.array[float],
    dt: float,
    contact_speculative_scale: float,
    contact_w: float,
    restitution_velocity_threshold: float,
    weighted: int,
    material: int,
):
    """The one writer on the sparse diagonal: a world's ``lanes`` threads stride over its rows (``dim = (worlds,
    lanes)``), every row (``counts``, the allocation's watermark) or its limit prefix alone (``prefix``:
    ``bounds[:, 1]``, where the compact boundary writes the contact rows in its producer), a row a lane: a contact row's
    records by the contact law, then each row as its own unit (:data:`read_row_unit`), walked by the one unit walker
    with the star (:data:`walk_slot_row`; the spin row a torque) and finished (:func:`slot_header`)."""
    world, lane = wp.tid()
    end = counts[world]
    if prefix != 0:
        end = bounds[world, 1]
    end = wp.min(end, support.units.shape[1])
    line = rows.first[world]
    for slot in range(lane, end, lanes):
        unit = read_row_unit(rows, data, law, JointSeeds(), line, slot, support.units[line, slot])
        if unit.rows > 0:
            unit.group = slots.dense_group[world]
            walk_slot_row(sink, unit)
            slot_header(
                rows, slots, data, v_hat, unit, dt, contact_speculative_scale, contact_w,
                restitution_velocity_threshold, weighted, material,
            )  # fmt: skip


def row_sink(factor, forest, placement, v_hat: wp.array) -> RowSink:
    """Every placement's sink (:class:`RowSink`) as far as they share it: the walk and its release, each articulation's
    response coordinates, the predictor ``v_hat``, and the rows' ``cfm``, incident velocities and diagonals."""
    sink = RowSink()
    walk = factor.walk_view()
    sink.walk, sink.release = walk, walk.release
    sink.response_dofs = forest.articulation_response_dof_count
    sink.v_hat, sink.cfm, sink.diag = v_hat, placement.options.pgs_cfm, placement.diag
    sink.row_incident = placement.row_incident
    return sink


def packed_sink(solver) -> RowSink:
    """The packed placements' sink (:class:`RowSink`, :func:`packed_walker`): where each side's entries sit in a packed
    row (:class:`.dense_rows.RowSupport`) and the entries, the general placement's packed rows, or the sparse diagonal's
    dense parts by size group (:data:`SLOT_ROWS`' ``Z``) with its star's coordinates."""
    placement = solver.placement
    sink = row_sink(solver.factor, solver.forest, placement, solver.v_hat)
    sink.layout = placement.row_support()
    if solver.routes.case == "K2":
        sink.entries = placement.J_by_size[placement.sparse_diagonal_dense_size]
        sink.star = int(placement.sparse_diagonal_response_size)
    else:
        sink.entries = placement.packed_Z
    return sink


# ---------------------------------------------------------------------------------------------------------------------
# The free bodies' rows: each side's six of a row's twelve, a packed row by the free bodies' own row
# ---------------------------------------------------------------------------------------------------------------------

walk_body_row = packed_walker(1, False, True, bodies=True)
"""The one unit walker into the free bodies' rows (:func:`packed_walker`): a row's sides from entry 0 and 6."""


@wp.kernel(enable_backward=False, launch_bounds=(256, 2))
def write_free_body_rows(sink: RowSink, data: ContactInput, total_threads: int, law: PatchRowLaw):
    """The free bodies' contact rows (``path`` 1, grid-stride over the contacts): each row's records by the contact law
    and the row as its own unit (:data:`read_row_unit`), walked by the one unit walker into the free bodies' rows
    (:data:`walk_body_row`; the spin row a torque), with its diagonal ``|z|^2 + cfm`` and incident velocity."""
    for c in range(wp.tid(), data.count[0], total_threads):
        slot = data.slot[c]
        if data.path[c] != 1 or slot < 0:
            continue
        line = sink.layout.first[data.world[c]]
        # The allocation's extent: recomputing eligibility can cross a floating-point threshold.
        rows = int(1)
        if data.slots_needed[c] >= 3:
            rows = data.slots_needed[c]
        for row in range(rows):
            source = unit_source(UNIT_CONTACT, 4 * c + row)
            walk_body_row(sink, read_row_unit(DenseRows(), data, law, JointSeeds(), line, slot + row, source))


@wp.kernel
def populate_rigid_velocity_limit_rows(
    bodies: FreeBodies,
    rows: DenseRows,
    sink: RowSink,
    rigid_velocity_limit_slot: wp.array[int],
):
    """Populate the free rigid bodies' speed rows, their linear and angular velocity limits.

    A speed row's Jacobian is the unit vector of the first DOF of the velocity it bounds (``e_0`` for the linear
    velocity, ``e_3`` for the angular one); the sweeps read the row's three DOFs from there, and its effective mass
    converts the speed correction into the impulse the row stores. Its diagonal is that axis walked as a velocity
    limit's covector at its coordinate (:data:`read_row_unit`, :data:`walk_body_row`), ``|L^-1 P e_k|^2 + cfm``; its
    ``Z`` then holds the axis unwhitened, and its incident velocity is zero: the speed law reads the body's velocity
    through its factor, not a row. ``mf_phi`` holds the limit.
    """
    candidate = wp.tid()
    body = bodies.free_rigid_body_indices[candidate]
    art = bodies.body_to_articulation[body]
    if art < 0:
        return
    if bodies.is_free_rigid[art] == 0:
        return

    world = bodies.art_to_world[art]
    if world < 0:
        return
    line = rows.first[world]

    base = RIGID_SPEED_ROWS * candidate
    for speed_axis in range(RIGID_SPEED_ROWS):
        slot = rigid_velocity_limit_slot[base + speed_axis]
        if slot < 0:
            continue
        limit = bodies.max_linear_velocity[body]
        if speed_axis == 1:
            limit = bodies.max_angular_velocity[body]

        row = sink.base + slot
        rows.sides[line, row] = wp.vec2i(art, -1)
        rows.row_type[line, row] = PGS_CONSTRAINT_TYPE_SPEED
        rows.row_parent[line, row] = -1
        rows.row_mu[line, row] = 0.0
        # The speed law's bias is its limit: the row's target is ``-limit``, as a joint velocity limit's, and its rhs
        # ``-target`` (:func:`.row_laws.row_rhs`); it has no gap.
        rows.row_beta[line, row] = 0.0
        rows.phi[line, row] = 0.0
        rows.target[line, row] = -limit

        k = 3 * speed_axis
        source = unit_source(UNIT_VELOCITY, 2 * (sink.walk.dof_start[art] + k))
        walk_body_row(sink, read_row_unit(rows, ContactInput(), PatchRowLaw(), JointSeeds(), line, row, source))
        for e in range(12):
            sink.entries[line, slot, e] = 0.0
        sink.entries[line, slot, k] = 1.0
        sink.row_incident[line, row] = 0.0


def free_body_sink(solver) -> RowSink:
    """The free bodies' sink (:class:`RowSink`, :data:`walk_body_row`): their rows ``mf_Z`` by their own row from the
    store's ``body_base``, each side's entries after the row's first side (the packed rows' layout, not dense)."""
    placement = solver.placement
    sink = row_sink(solver.factor, solver.forest, placement, solver.v_hat)
    layout = placement.row_support()
    layout.dense = 0
    sink.layout, sink.entries, sink.base = layout, placement.mf_Z, int(placement.body_base)
    return sink

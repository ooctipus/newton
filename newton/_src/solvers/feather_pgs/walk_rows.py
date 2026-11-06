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
- **the general placement** (case K6, FL3.3): each side's entries in the packed rows, through each tree's class's path
  table (:func:`packed_walker`), world-major with the bilateral blocks' closure (:func:`walked_rows_kernel`);
- **the sparse diagonal** (case K2, FL3.5): the arm's side into the row's dense part, a key of the star one step to
  the row's slot (:func:`slot_rows_kernel`, :func:`slot_header`);
- **the free bodies' rows** (the packed rows of the line from ``body_base``, each side six entries): a
  contact row, or a speed row's axis, a covector at the body's root, for its diagonal, one writer over both
  (:func:`write_free_body_rows`), through the packed walker (:func:`packed_walker`).

Each row's header is its diagonal ``|z|^2 + cfm`` (the walk's sums of squares, re-read where two seeds share a side or
a side slips), its incident velocity ``J v_hat`` (the step's ``J`` entries each walk passes) and its rhs by the row law
(:func:`.row_kernels.row_bias`). A wrench enters as the step's ``J`` (the step's columns the finish published, shifted to
the walk's frames, :attr:`.factor.walk.WalkTrees.now`), so a step between refreshes whitens the step's ``J`` by the
held factor. A slipping drive's ``K`` leaves the rows by its release along its path (``R``, the predictor's border,
:func:`.tree_kinetic_state.border_solve`), which keeps every row inside its support.

No ``J`` is formed. The rows and the sweeps' leave share the walk's one frame (FL3_design §1.2); the sweeps' change
goes to the step's one leave (:func:`.factor.walk.walk_leave_kernel`).
"""

from __future__ import annotations

from functools import cache
from types import MappingProxyType
from typing import NamedTuple

import warp as wp

from ...core.execution import Buckets, Work, bucket_item, shape
from ...core.memory import built_for
from . import field_group
from .contact_input import ContactFrame, ContactInput, contact_frame
from .contact_laws import CONTACT_SPIN_ROW, contact_row_entry, contact_seed, spin_seed
from .device_preludes import DevicePrelude, dual_native, lane_group, substitute
from .factor.closure import close_block, close_row
from .factor.fields import Preelim
from .factor.walk import (
    FREE_BODY_WALK,
    WALK_COORDINATE,
    WALK_FRAME,
    WALK_LINK,
    WalkRelease,
    WalkTrees,
    walk_prelude,
    walk_row_prelude,
    walk_tables,
)
from .field_group import Field
from .friction_patches import PatchRowLaw
from .joint_table import FreeBodies
from .kernel_template import python_kernel
from .projection_laws import friction_ball_metric, friction_pair_metric
from .row_build import write_contact_row
from .row_kernels import RIGID_SPEED_ROWS, RowBias, row_bias
from .row_laws import LINK_ROWS
from .row_records import (
    UNIT_CONTACT,
    UNIT_LIMIT,
    UNIT_LINK,
    UNIT_MIMIC,
    UNIT_SHIFT,
    UNIT_SPEED,
    UNIT_VELOCITY,
    DenseRows,
    RowSupport,
    unit_source,
)
from .sparse_mass_matrix import _SparseRowSupport
from .tree_scan import WALK_BOTH, held_fits, record_half

__all__ = [
    "SUPPORT_UNIT_LANES",
    "WRITE_REST",
    "WRITE_ROWS",
    "WRITE_TOPOLOGY",
    "JointSeeds",
    "RowSink",
    "RowUnit",
    "SlotRows",
    "UnitList",
    "bind_slot_rows",
    "free_body_sink",
    "packed_sink",
    "packed_walker",
    "row_sink",
    "slot_rows_kernel",
    "slot_writer",
    "support_rows_kernel",
    "support_threads",
    "support_units_kernel",
    "unit_list",
    "unit_reader",
    "walked_rows_kernel",
    "write_free_body_rows",
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


WRITE_ROWS, WRITE_TOPOLOGY, WRITE_REST = wp.constant(0), wp.constant(1), wp.constant(2)
"""The one writer's parts (:func:`walked_rows_kernel`'s ``part``): every row, the rows a world's topology fixes, and the
rest with the closure."""

WRITER_BLOCK_ROWS = 1
"""The rows a lane of the topology part walks as one block (:func:`block_walker`): one, its seeds in one sweep. A lane
already walks beside its warp's others for free; more rows a lane serialize them into its instruction stream, and the
writer turns issue-bound (ZK7, franka at 16,384 worlds: the part 39 us at one row a lane, 55 at two, 58 at four)."""

WALKER_LINE = 80
"""The one walker's register line: a world-lane writer of the packed rows or the star's slots holds its walk at 80
registers, 24 warps a multiprocessor, for a latency-bound walk whose speed follows its resident warps. Unbound, the
class tables' walk takes 84-86 registers, past the granule: 16 warps (franka-cabinet 7% slower, gate_fast)."""


class WriterShape(NamedTuple):
    """One writer's iteration shape (a row of :data:`WRITERS`): its block and the register line its walk holds. Its
    launch bound is the block at the line, the blocks a multiprocessor keeps there, and its register budget a lane the
    line the bound leaves (:class:`.core.execution.Work`, for the architecture the build is sized for)."""

    block: int
    line: int

    def _shape(self):
        return shape(Work(0, block=self.block, registers=self.line), built_for(None))

    @property
    def bounds(self) -> tuple[int, int]:
        """The launch bound ``(block, blocks a multiprocessor)``."""
        return self._shape().bounds

    @property
    def registers(self) -> int:
        """The register budget a lane the bound leaves."""
        return self._shape().registers


WRITERS = MappingProxyType(
    {
        "packed": WriterShape(128, WALKER_LINE),
        "packed_blocks": WriterShape(128, 96),
        "slots": WriterShape(256, WALKER_LINE),
        "support_units": WriterShape(256, 64),
        "support": WriterShape(128, 168),
        "free_body": WriterShape(256, 128),
    }
)
"""Every writer's iteration shape, by sink: what its lanes take, its block and its register line.

- **World lanes**, a world's lane group striding over its rows, a row a lane: the general writer's packed rows
  (:func:`walked_rows_kernel`; a group of :func:`writer_lanes`, whole groups a block of 128) at the walker's line; with
  bilateral blocks (``packed_blocks``) at 96, since its link rows hold a second point across the walk and its closure
  follows it (97 unbound; at the walker's line it spilled 88 B a thread, sm_120); the star's slots
  (:func:`slot_rows_kernel`, a block a world) at the walker's line; the support placement's unit lister
  (:func:`support_units_kernel`) at 64, where its third kind's reservation would lift it to 66 (sm_120, ptxas), a
  granule that would cost it a block.
- **The step's unit list**, two waves of the line over the GPU (:func:`support_threads`): the support writer
  (:func:`support_rows_kernel`) at 168, where its four-row walker and the friction triple's metric hold 164 with no
  spill (sm_120, ur10-reach's class); at 128 they spilled 144 B a lane and loaded 412 B back, every unit's walk paid
  the spills' local-memory traffic, and the kinetic finish two launches later paid its L2 lines. Measured
  (``SP/core/FL/zwr8``, L5): franka-reach 4.3%, ant 1.3%, anymald 1.1% and ur10-reach 0.4% faster on the PRO.
- **The free bodies' units**, their contacts then their speed rows, a unit a thread, two waves of the line: the
  free-body writer (:func:`write_free_body_rows`) at 128.
"""


@cache
def unit_reader(links: bool, unit_rows: int = 1, contacts: bool = True) -> wp.Function:
    """Every placement's unit reader (:func:`row_unit`). A contact's unit is its rows from its first, at most
    ``unit_rows``, from the contact's frame: the support placement's, whose unit list holds a contact's rows as one unit
    and whose first launch wrote their records (:func:`support_units_kernel`). With ``unit_rows`` 1 it is the row
    alone, its records written here by the contact law: the world-major writers', a lane a row (the producer's, folded
    into the writer). ``links``: with the link rows' seeds, where the writer closes bilateral blocks (the general
    placement's block variants; the sparse diagonal and the support placement admit none); ``contacts``: with the
    contact rows' (not the topology part's, :data:`WRITE_TOPOLOGY`)."""

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
        if wp.static(contacts) and kind == UNIT_CONTACT:
            c = index // 4
            frame = contact_frame(data, law, c)
            if wp.static(unit_rows > 1):
                unit.row = index - 4 * c
                unit.rows = wp.min(data.slots_needed[c], wp.static(unit_rows))
                unit.frame = wp.matrix_from_rows(frame.normal, frame.tangent0, frame.tangent1)
                unit.friction_point = frame.friction_point
                unit.art_a = data.art0[c]
                unit.body_a = frame.body_a
                unit.art_b = data.art1[c]
                unit.body_b = frame.body_b
                unit.point = frame.point
                unit.point_b = frame.point
                if unit.row == CONTACT_SPIN_ROW:
                    unit.torque = 1
            else:
                unit = contact_row_unit(data, law, frame, c, index - 4 * c, line, slot)
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
    name += "" if contacts else "_topology"
    row_unit.__name__ = row_unit.__qualname__ = name
    return wp.func(row_unit)


@cache
def walked_rows_kernel(blocks: int, mimic_blocks: bool, side: bool, tables: tuple, block: int = 0) -> wp.Kernel:
    """The one writer of the general placement, world-major: a world's group of ``lanes`` lanes (:func:`writer_lanes`)
    writes its rows in the block primitive's phases (FL3_design §1.4, W9), the group synchronized between them:

    1. *rows*: a lane a row, every row of the world: a contact row's records by the contact law, then the row as its own unit
       (:func:`unit_reader`), walked by the one unit walker with the packed rows as its sink
       (:func:`packed_walker`; the spin row a torque), each side's slips released by the
       predictor's border's ``R`` (:func:`.tree_kinetic_state.border_solve`), then its rhs by the row law
       (:func:`.row_kernels.row_bias`). A row a lane, not a contact a lane: the world's lanes stay busy (the support
       placement's unit list spreads a contact's unit over the GPU instead, :func:`support_rows_kernel`);
    2. *closure* (``blocks``, the largest bilateral block's rows; 0 for none): the lanes close each of the world's
       blocks (:func:`.factor.closure.close_block`; its rows are the links', and the mimics' with
       ``mimic_blocks``), then correct every other row on the blocks its sides touch
       (:func:`.factor.closure.close_row`).

    The closure's seed stays a launch after the warm start's gather (:meth:`.tree_factor.TreeFactor.seed_blocks`).
    One launch with group barriers, not a second launch for the block phase: FL33D's measurement (``fl33d/``).

    ``part`` (:data:`WRITE_ROWS`, :data:`WRITE_TOPOLOGY`, :data:`WRITE_REST`): the rows a world's topology fixes (its
    links, mimics, limits and velocity limits, the slots below its family bound ``bounds[world, 1]``,
    :meth:`.rows.RowLaws.bound_phase`) are known before the contacts, so a launch can write them alone, beside the
    contact pass, and a second the rest and the closure; one launch writes every row where the rows' order holds no
    family bound (keyed rows).

    ``side``: whether the model's records can have a side (:attr:`.tree_factor.TreeFactor.walk_side`); ``tables``: the
    walking classes' path tables (:func:`.factor.walk.walk_prelude`), each tree's path read from constant memory by its
    class.

    ``block`` (the topology part's kernel, :data:`WRITE_TOPOLOGY`; 0 for the others): a lane takes ``block``
    consecutive rows and walks its covector rows (limits, velocity limits, mimics) as one block, each tree's in one
    sweep (:func:`block_walker`), a link row by the unit walker.

    The worlds: every world, or with ``claimed`` the worlds of the claims' ``walked`` buckets
    (:class:`~newton._src.core.execution.Buckets`, the claim stage's: :func:`.sweep_levels.claim_worlds`), a world a
    lane group of the launch, where the keyed Jacobi tiers walk their own worlds' seeds in their kernels."""
    has_blocks = blocks > 0
    capacity = max(blocks, 1)
    mimics = int(mimic_blocks)
    # A link row is a block row (block_row; links plan blocks, TreeFactor.plan_blocks): the writer reads link rows
    # where it closes blocks. Elsewhere their seeds would hold a second point across every walk.
    read_unit = unit_reader(has_blocks, contacts=not block)
    walk = packed_walker(1, False, side, tables)
    if block:
        walk_block = block_walker(block, side, tables)
        ints, floats = wp.types.vector(block, int), wp.types.vector(block, float)

    def kernel(
        rows: DenseRows,
        support: RowSupport,
        counts: wp.array[int],
        walked: Buckets,
        claimed: int,
        lanes: int,
        data: ContactInput,
        law: PatchRowLaw,
        sink: RowSink,
        preelim: Preelim,
        mimic_slot: wp.array[int],
        link_slot: wp.array[int],
        seeds: JointSeeds,
        bias: RowBias,
        bounds: wp.array2d[int],
        part: int,
    ):
        world, lane = wp.tid()
        if claimed != 0:
            # The claims' worlds: a lane group past them returns whole, before any barrier.
            world = bucket_item(walked, world)
            if world < 0:
                return
        # The writer's rows are the line's dense rows: the packed rows past them are the free bodies'.
        dense_rows = support.units.shape[1]
        count = wp.min(counts[world], dense_rows)
        line = rows.first[world]
        # The part's rows: the topology's below the family bound (the count is not final before the contact pass),
        # the rest above it.
        first = int(0)
        end = count
        if part == WRITE_TOPOLOGY:
            end = wp.min(bounds[world, 1], dense_rows)
        elif part == WRITE_REST:
            first = wp.min(bounds[world, 1], count)
        if wp.static(block > 0):
            for start in range(first + lane * wp.static(block), end, lanes * wp.static(block)):
                arts, dof0, dof1 = ints(), ints(), ints()
                value0, value1 = floats(), floats()
                for r in range(wp.static(block)):
                    dof0[r] = -1
                    if start + r < end:
                        unit = read_unit(rows, data, law, seeds, line, start + r, support.units[line, start + r])
                        if unit.dof0 >= 0:
                            arts[r] = unit.art_a
                            dof0[r] = unit.dof0
                            value0[r] = unit.value0
                            dof1[r] = unit.dof1
                            value1[r] = unit.value1
                        elif wp.static(has_blocks):
                            walk(sink, unit)
                rows_here = wp.min(end - start, wp.static(block))
                walk_block(sink, line, start, rows_here, arts, dof0, value0, dof1, value1)
                for slot in range(start, start + rows_here):
                    row_bias(rows, bias, line, slot, sink.row_incident[line, slot])
            return
        for slot in range(first + lane, end, lanes):
            source = support.units[line, slot]
            unit = read_unit(rows, data, law, seeds, line, slot, source)
            if unit.rows > 0:
                walk(sink, unit)
                incident = sink.row_incident[line, slot]
                if source >> UNIT_SHIFT == UNIT_SPEED:
                    incident = speed_axis(sink, line, slot, unit.art_a, unit.dof0)
                row_bias(rows, bias, line, slot, incident)
        if wp.static(has_blocks):
            # The closure reads every row of the world: the launch that writes the last of them closes.
            if part != WRITE_TOPOLOGY:
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
    name += f"_topology{block}" if block else ""
    return python_kernel(kernel, name, launch_bounds=WRITERS["packed_blocks" if has_blocks else "packed"].bounds)


def writer_kernel(solver, part: int = WRITE_ROWS) -> wp.Kernel:
    """The one writer's variant for the solver's class (:func:`walked_rows_kernel`), from its topology: the block phase
    where bilateral blocks exist (mimics among them under ``enable_bilateral_preelimination``), the records' side
    where a walked joint can have two halves (:attr:`.tree_factor.TreeFactor.walk_side`), and the walking classes'
    paths (:attr:`.tree_factor.TreeFactor.walk_classes`); the topology ``part``'s, its rows walked in blocks of
    :data:`WRITER_BLOCK_ROWS`."""
    factor = solver.factor
    blocks = factor.block_capacity if factor.eliminates else 0
    mimics = bool(blocks and solver.options.enable_bilateral_preelimination)
    block = WRITER_BLOCK_ROWS if part == WRITE_TOPOLOGY else 0
    return walked_rows_kernel(blocks, mimics, factor.walk_side, walk_tables(factor.walk_classes), block)


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
    // A friction unit's Gram couplings: its pair's with its normal and each other, and its spin row's where it has one
    // (a unit of three rows leaves its fourth zero, whose couplings stay zero).
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
            if (!add && rows >= 3) gram();
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
            if (rows >= 3) gram();
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

_SUPPORT_RECORD = """    // A friction unit's block record from its Gram (sparse_contact.CONTACT_RECORD_FLOATS): its couplings W_nt1,
    // W_nt2, W_t1t2 and 1 / W_nn, then the spin row's couplings W_ns, W_t1s, W_t2s where it has a spin row or,
    // written by the kernel from these, the tangent pair's metric (projection_laws.friction_pair_metric).
    if (rows >= 3) {
        float* record = sink.records.data + ((size_t)unit.line * sink.records.shape[1] + unit.slot / 3)
                        * sink.records.shape[2];
        ST(record + 0, cross[0]);
        ST(record + 1, cross[1]);
        ST(record + 2, cross[2]);
        ST(record + 3, 1.0f / (norm[0] + sink.cfm));

    }
"""
_SPIN_RECORD = """    // A contact with a spin row: its friction rows' couplings W_t1t2, W_t1s and W_t2s, which the kernel replaces by
    // the friction triple's metric (sparse_contact.CONTACT_RECORD_FLOATS, projection_laws.friction_ball_metric).
    if (rows == 4) {
        float* record = sink.records.data + ((size_t)unit.line * sink.records.shape[1] + unit.slot / 3)
                        * sink.records.shape[2];
        ST(record + 0, cross[2]);
        ST(record + 1, cross[4]);
        ST(record + 2, cross[5]);
    }
"""


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

SUPPORT_REGISTERS = WRITERS["support"].registers
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
        record = _SPIN_RECORD if spin else _SUPPORT_RECORD
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
    walk = walk_prelude(((parent, first),), rows, half)
    return dual_native(body, preludes=(walk, walk_row_prelude(rows), _support_table(node)))


@wp.struct
class RowSink:
    """What a placement's writer reads and writes besides a unit's seeds, every placement's (:data:`_SUPPORT`): the walk
    and its release, each articulation's response coordinates, the predictor ``v_hat``, and the rows' ``cfm``, entries,
    incident velocities and diagonals; the support placement's (case K1, :func:`support_rows_kernel`) support table,
    each body's and coordinate's support mask (factor coordinates), each articulation's first world coordinate, and the
    rows' coordinates and block records; the packed rows' layout, where each side's entries sit in a packed row
    (:class:`.dense_rows.RowSupport`, :func:`packed_walker`), the star's coordinates (the sparse diagonal's), and the
    free bodies' first row in the store's line (``base``: their slots count from it). A placement binds what
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


@wp.func
def contact_row_unit(
    data: ContactInput, law: PatchRowLaw, frame: ContactFrame, c: int, row: int, line: int, slot: int
) -> RowUnit:
    """Contact ``c``'s row ``row`` as its own unit at row ``slot``, from the contact's front end ``frame``: its records
    by the contact law (:func:`.row_build.write_contact_row`), its bodies, direction and point (the spin row a torque
    about the normal)."""
    body_a, body_b, direction, point = write_contact_row(data, law, frame, c, row, False)
    unit = RowUnit()
    unit.line = line
    unit.slot = slot
    unit.rows = 1
    unit.row = row
    unit.frame = wp.matrix_from_rows(direction, direction, direction)
    unit.art_a = data.art0[c]
    unit.body_a = body_a
    unit.art_b = data.art1[c]
    unit.body_b = body_b
    unit.point = point
    unit.point_b = point
    unit.friction_point = point
    unit.dof0 = -1
    unit.dof1 = -1
    if row == CONTACT_SPIN_ROW:
        unit.torque = 1
    return unit


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

_PACKED_SIDE = """$skip$tree
        int seeds = 0;
        if (covector) {
            seeds = dof1 >= 0 ? 2 : 1;
        } else {
            if (body_a >= 0 && art_a == art) ++seeds;
            if (body_b >= 0 && art_b == art) ++seeds;
        }
        if (seeds == 0) continue;
        // The side's first entry (row_records.RowSupport): in a packed row the articulation's first world coordinate
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


_CLASS_TREE = """        // The tree's class: its path from the classes' tables in constant memory.
        WALK_TREE(RD(walk.cls.data + art));"""


@cache
def packed_walker(rows: int, spin: bool, side: bool, tables: tuple, star: bool = False) -> wp.Function:
    """The general placement's unit walker (:data:`_SUPPORT`, the support writer's text with the packed rows as its
    sink): a unit's ``rows`` rows walked together through its tree's path, each footprint stored at its coordinate's entry
    in the side's packed row, its diagonal ``|z|^2 + cfm`` and incident ``J v_hat`` from the same walk; ``spin``: the
    unit's fourth row is its spin row; ``side``: whether records can have a side. With ``star``, the sparse diagonal's:
    the arm's side into the row's dense part (its size group's row, ``unit.group``), a key's side left to the writer
    (:func:`star_step`), and the diagonal stored without its slots' squares and ``cfm``, which the writer adds.
    ``tables``: the walking classes' path tables (:func:`.factor.walk.walk_prelude`),
    fixed at build, each tree's path read from constant memory by its class (one table: every tree's)."""
    tree = _CLASS_TREE if len(tables) > 1 else ""
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
            tree=tree,
            base="0" if star else _PACKED_BASE,
        ),
        rank="base + c",
        below="",
        wrench=substitute(_PACKED_WRENCH, spin=int(spin)),
        close="",
        entries_end="S",
        zrow="(unit.group * sink.entries.shape[1] + unit.slot)" if star else "row0",
        diag="norm[r]" if star else _HEADER_DIAG,
        # RD a plain load here, not the support writer's non-coherent __ldg: the K2 writer ran 1.4% faster on sm_120.
        rd="*p",
    )
    half = record_half(side)
    walk = walk_prelude(tables, rows, half)
    tag = ("_spin" if spin else "") + walk.name.removeprefix(f"walk_{rows}")
    tag += "_star" if star else ""
    snippet = dual_native(body, preludes=(walk, walk_row_prelude(rows)))
    return unit_walker(f"packed_walk_{rows}{tag}", snippet)


_BLOCK = """    // A block of up to $rows covector rows of the packed sink from row slot (walk_rows.block_walker): each row's entries
    // cleared, then each tree's rows walked together (WALK_SWEEP), a walk a seed (a mimic's second its own, adding
    // where the two meet), then each row's slips released, its diagonal |z|^2 + cfm and its incident J v_hat. A row's
    // arithmetic is the unit walker's (_SUPPORT) on the row alone.
#if defined(__CUDA_ARCH__)
    const auto ST = [](auto* p, const auto v) {
        __builtin_assume(__isGlobal(p));
        *p = v;
    };
    const auto RW = [](const float* p) -> float {
        __builtin_assume(__isGlobal(p));
        return *p;
    };
#else
    const auto ST = [](auto* p, const auto v) { *p = v; };
    const auto RW = [](const float* p) -> float { return *p; };
#endif
    const auto& walk = sink.walk;
    const auto& release = sink.release;
    const int S = sink.entries.shape[2];
    float* const Z = sink.entries.data + ((size_t)line * sink.entries.shape[1] + slot) * S;
    const int rec0 = line * sink.diag.shape[1] + slot;
    float norm[$rows], incident[$rows];
    int pending = 0, reread = 0;
WALK_UNROLLED
    for (int r = 0; r < $rows; ++r) {
        norm[r] = incident[r] = 0.0f;
        if (r >= count || dof0[r] < 0) continue;
        pending |= 1 << r;
        for (int e = 0; e < S; ++e) ST(Z + r * S + e, 0.0f);
    }
    // The block's trees in turn: the first row still pending names the next.
#pragma unroll 1
    while (pending != 0) {
        int art = -1, mine = 0;
WALK_UNROLLED
        for (int r = 0; r < $rows; ++r)
            if (art < 0 && ((pending >> r) & 1) != 0) art = arts[r];
WALK_UNROLLED
        for (int r = 0; r < $rows; ++r)
            if (((pending >> r) & 1) != 0 && arts[r] == art) mine |= 1 << r;
        pending &= ~mine;
        if (*(sink.response_dofs.data + art) <= 0) continue;
        const int j0 = walk.joint_start.data[art], d0 = walk.dof_start.data[art];
$tree
        // A covector row's one side: its tree's first world coordinate (dense) or entry 0.
        const int base = sink.layout.dense != 0 ? sink.layout.offset.data[art] : 0;
        const float* data = walk.data.data + (size_t)d0 * $coordinate;
        const float* link = walk.link.data + (size_t)j0 * $link;
        const float* vh = sink.v_hat.data + d0;
        // Walk 2 r a row's first seed, 2 r + 1 its second (a mimic's); a row of two adds where they meet.
        int at[2 * $rows], body[2 * $rows], adds = 0;
        float seed[2 * $rows];
WALK_UNROLLED
        for (int r = 0; r < $rows; ++r) {
            const bool on = ((mine >> r) & 1) != 0;
            at[2 * r] = on ? dof0[r] - d0 : -1;
            at[2 * r + 1] = on && dof1[r] >= 0 ? dof1[r] - d0 : -1;
            seed[2 * r] = value0[r];
            seed[2 * r + 1] = value1[r];
            if (at[2 * r + 1] >= 0) adds |= 1 << r;
            body[2 * r] = at[2 * r] >= 0 ? walk.dof_body.data[d0 + at[2 * r]] : -1;
            body[2 * r + 1] = at[2 * r + 1] >= 0 ? walk.dof_body.data[d0 + at[2 * r + 1]] : -1;
        }
        reread |= adds;
        WALK_SWEEP(data, link, at, body, seed, [&](const int w, const int c, const float z) {
            const int r = w >> 1;
            float* const entry = Z + r * S + base + c;
            if (((adds >> r) & 1) != 0) {
                ST(entry, RW(entry) + z);
            } else {
                ST(entry, z);
                norm[r] += z * z;
            }
        });
WALK_UNROLLED
        for (int k = 0; k < 2 * $rows; ++k)
            if (at[k] >= 0) incident[k >> 1] += seed[k] * vh[at[k]];
        // A slipping tree's release (the predictor's border) on each of its rows (WALK_RELEASE_ON).
        if (release.count.data[art] > 0) {
            reread |= mine;
WALK_UNROLLED
            for (int r = 0; r < $rows; ++r)
                if (((mine >> r) & 1) != 0)
                    WALK_RELEASE_ON(release, art, Z + r * S, 1, S, [&](const int c) -> int { return base + c; });
        }
    }
WALK_UNROLLED
    for (int r = 0; r < $rows; ++r) {
        if (r >= count || dof0[r] < 0) continue;
        if (((reread >> r) & 1) != 0) {
            norm[r] = 0.0f;
            for (int e = 0; e < S; ++e) {
                const float zc = RW(Z + r * S + e);
                norm[r] += zc * zc;
            }
        }
        ST(sink.diag.data + rec0 + r, norm[r] + sink.cfm);
        ST(sink.row_incident.data + rec0 + r, incident[r]);
    }"""


@cache
def block_walker(rows: int, side: bool, tables: tuple) -> wp.Function:
    """The general placement's walker of a block of up to ``rows`` covector rows (:data:`_BLOCK`): a world's rows at
    consecutive slots from ``slot`` (``count`` of them; a row of ``dof0`` -1 is not the block's), each row's seeds its
    ``value0 e_dof0`` and, a mimic's, ``value1 e_dof1`` on articulation ``arts``. Each tree's rows walk one sweep of its
    bodies (``WALK_SWEEP``, the walk with entry points: each seed enters at its coordinate), so a world's rows run as
    independent chains of one sweep, not one chain a lane; each row's footprints, diagonal and incident are the unit
    walker's (:func:`packed_walker`). ``side`` and ``tables`` as :func:`packed_walker`'s."""
    walk = walk_prelude(tables, 2 * rows, record_half(side))
    body = substitute(
        _BLOCK,
        rows=rows,
        coordinate=WALK_COORDINATE,
        link=WALK_LINK,
        tree="        WALK_TREE(walk.cls.data[art]);" if len(tables) > 1 else "",
    )
    ints, floats = wp.types.vector(rows, int), wp.types.vector(rows, float)

    def native(sink, line, slot, count, arts, dof0, value0, dof1, value1): ...

    # The block's vectors are sized at build: the signature's types are set here, not evaluated from its text.
    native.__annotations__ = {
        "sink": RowSink,
        "line": int,
        "slot": int,
        "count": int,
        "arts": ints,
        "dof0": ints,
        "value0": floats,
        "dof1": ints,
        "value1": floats,
    }
    name = f"block_walk_{rows}" + walk.name.removeprefix(f"walk_{2 * rows}")
    native.__name__ = native.__qualname__ = name
    return wp.func_native(dual_native(body, preludes=(walk,)))(native)


SUPPORT_UNIT_LANES = 16
SUPPORT_UNIT_LANES = 16
"""A world's lane group in the support writer's first launch (:func:`support_units_kernel`): its rows' records and
units, a lane a row; the walkers' lanes (:func:`support_threads`) do not depend on it."""

UNIT_BUCKETS = 16
"""The unit list's buckets (:class:`UnitList`): a world's units go to bucket ``world % UNIT_BUCKETS``, so the lane
groups' reservations spread over that many counters of each kind."""


@wp.struct
class UnitList:
    """The step's units, one fixed slot each (:func:`support_units_kernel` appends, :func:`support_rows_kernel`
    walks), listed by kind so a warp of walkers runs one walker: units of four rows (a contact with its spin row),
    units of three (a contact's normal and pair) and units of one row. A unit is a row's seed and the rows it walks
    together, at least one row, so a world's units fit its rows' ``capacity`` and its units of four a quarter of it:
    the list holds every world's at those bounds and cannot overflow. Bucket ``b`` holds its worlds' units in
    ``span`` slots, the units of three from the front and the one-row units from the back, then ``quad`` slots of
    units of four (none where no spin rows appear), each as ``line * capacity + slot`` of its first row;
    ``count[3 b]``, ``count[3 b + 1]`` and ``count[3 b + 2]`` are its counts of three, one and four rows, zeroed
    before the step's append. The order within a kind is the lane groups' arrival, on which no unit's rows depend."""

    entry: wp.array[int]
    count: wp.array[int]
    span: int
    quad: int
    capacity: int


def unit_list(worlds: int, capacity: int, device, *, spin_rows: bool) -> UnitList:
    """An empty :class:`UnitList` for ``worlds`` worlds of ``capacity`` rows each; with ``spin_rows`` (the laws let
    contacts carry spin rows) its region of units of four."""
    units = UnitList()
    units.capacity = int(capacity)
    per_bucket = -(-int(worlds) // UNIT_BUCKETS)
    units.span = per_bucket * int(capacity)
    units.quad = per_bucket * (int(capacity) // 4) if spin_rows else 0
    units.entry = wp.empty(max(UNIT_BUCKETS * (units.span + units.quad), 1), dtype=int, device=device)
    units.count = wp.zeros(3 * UNIT_BUCKETS, dtype=int, device=device)
    return units


_UNIT_APPEND = """    // This round's units of the world's lane group into its bucket: one reservation a kind for the group.
    const bool wide = rows > 1 && rows < 4;
    const unsigned wide_votes = BALLOT(wide), one_votes = BALLOT(rows == 1), quad_votes = BALLOT(rows == 4);
    const int wide_first = RESERVE(units.count.data + 3 * bucket, COUNT(wide_votes));
    const int one_first = RESERVE(units.count.data + 3 * bucket + 1, COUNT(one_votes));
    int* const region = units.entry.data + (size_t)bucket * (units.span + units.quad);
    if (wide) region[wide_first + BELOW(wide_votes)] = entry;
    if (rows == 1) region[units.span - 1 - (one_first + BELOW(one_votes))] = entry;
    if (quad_votes != 0) {
        const int quad_first = RESERVE(units.count.data + 3 * bucket + 2, COUNT(quad_votes));
        if (rows == 4) region[units.span + quad_first + BELOW(quad_votes)] = entry;
    }"""


@wp.func_native(dual_native(_UNIT_APPEND, preludes=(lane_group("lanes"),)))
def unit_append(lanes: int, units: UnitList, bucket: int, rows: int, entry: int):
    """Append this lane's unit of ``rows`` rows (0 for none) to bucket ``bucket`` of ``units``
    (:class:`UnitList`); every lane of the world's group of ``lanes`` calls it together."""


@wp.func
def unit_total(units: UnitList) -> int:
    """The list's units over its buckets."""
    total = int(0)
    for k in range(3 * UNIT_BUCKETS):
        total += units.count[k]
    return total


@wp.func
def unit_entry(units: UnitList, item: int) -> int:
    """The list's ``item``-th unit: its units of four first, then of three, bucket by bucket, then its one-row
    units."""
    quads = int(0)
    wide = int(0)
    for b in range(UNIT_BUCKETS):
        quads += units.count[3 * b + 2]
        wide += units.count[3 * b]
    kind = int(2)
    rank = item
    if item >= quads:
        kind = 0
        rank = item - quads
        if rank >= wide:
            kind = 1
            rank -= wide
    b = int(0)
    n = units.count[kind]
    while rank >= n:
        rank -= n
        b += 1
        n = units.count[3 * b + kind]
    at = rank
    if kind == 1:
        at = units.span - 1 - rank
    elif kind == 2:
        at = units.span + rank
    return units.entry[b * (units.span + units.quad) + at]


@cache
def support_units_kernel(max_rows: int, lanes: int) -> wp.Kernel:
    """The support placement's units (case K1, the writer's first launch): ``dim = (worlds, lanes)``, a world's lane
    group of ``lanes`` striding over its rows, a lane a row: each contact row's records by the contact law
    (:func:`.row_build.write_contact_row`, its unit source written where its slot is reserved,
    :func:`.row_slots.allocate_world_contact_slots`), and each unit's first row appended to the step's
    :class:`UnitList`: a contact's ``min(slots_needed, max_rows)`` rows as one unit, a joint row (a limit, a velocity
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
            unit_rows = int(0)
            if slot < count:
                source = sources[line, slot]
                kind = source >> UNIT_SHIFT
                index = source & ((1 << UNIT_SHIFT) - 1)
                if kind == UNIT_CONTACT:
                    c = index // 4
                    row = index - 4 * c
                    write_contact_row(data, law, contact_frame(data, law, c), c, row, False)
                    if row == 0:
                        unit_rows = wp.min(data.slots_needed[c], wp.static(max_rows))
                elif kind == UNIT_LIMIT or kind == UNIT_VELOCITY or kind == UNIT_MIMIC:
                    unit_rows = 1
            unit_append(wp.static(lanes), units, bucket, unit_rows, line * units.capacity + slot)

    return python_kernel(kernel, f"support_units_{max_rows}_{lanes}", launch_bounds=WRITERS["support_units"].bounds)


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
    striding over the step's :class:`UnitList` (:func:`support_units_kernel`) kind by kind, units of four rows, of
    three, then of one, so a warp runs one walker. A lane walks a unit: a contact's rows from its first, a joint row alone. The generator form (Gamma per body,
    ``z = Gamma w``) measured slower on g1 than this direct walk, so the writer walks each unit.

    The class's tree (``parent``, ``first``: :func:`.factor.walk.walk_prelude`'s table) and each coordinate's factor
    coordinate ``node`` (the support masks' bit) are baked into constant memory. A contact unit of three rows (four
    with its spin row, where ``torsion`` lets spin rows appear; each count its own walker) is walked together and its Gram writes its block record where ``records`` (the
    triplet sweeps'); a unit of one row (a normal alone, a limit, a velocity limit, a mimic) by the one-row walker. The
    contact rows' records are the first launch's; a joint row's records and unit source are its producer's (the limit,
    velocity-limit and mimic producers). The walker stores its footprints at their ranks in its support
    (``row_factor``, with the coordinates ``row_dof`` or the support class ``row_class``), its diagonal ``|z|^2 + cfm``,
    its incident ``J v_hat`` and its rhs by the row law (:func:`.row_kernels.row_bias`). ``half`` is the class's records'
    half (:func:`.tree_scan.classes_half`), compiled into its walks.
    """
    unit_rows = 4 if torsion else 3

    def walker(rows: int, with_records: bool):
        snippet = _support_snippet(parent, first, node, rows, with_records, torsion and rows == 4, half)
        return unit_walker(f"support_walk_{rows}", snippet)

    walk_one = walker(1, False)
    walk_unit = walker(unit_rows, records)
    # Where spin rows may appear, a contact without one walks three rows, not the spin layout's four.
    walk_three = walker(3, records) if torsion else walk_unit
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
        bias: RowBias,
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
            elif wp.static(torsion) and unit.rows == 3:
                walk_three(sink, unit)
            else:
                walk_unit(sink, unit)
            if wp.static(records):
                if unit.rows == 3:
                    # The friction pair's metric of the diagonals as written, which the triplet sweeps reuse (a spin
                    # row's couplings take its place in a unit of four).
                    k = slot // 3
                    coupling = sink.records[line, k, 2]
                    largest, smallest, axis = friction_pair_metric(
                        sink.diag[line, slot + 1], coupling, sink.diag[line, slot + 2]
                    )
                    sink.records[line, k, 4] = largest
                    sink.records[line, k, 5] = smallest
                    sink.records[line, k, 6] = axis[0]
                    sink.records[line, k, 7] = axis[1]
                elif wp.static(torsion) and unit.rows == 4:
                    # The friction triple's metric of the block as written, formed once per solve: the sweeps'
                    # visits only project (projection_laws.friction_triple_from_metric). Its eigenvalues, then its
                    # eigenbasis as a rotation, take the triple's couplings' place.
                    k = slot // 3
                    t12 = sink.records[line, k, 0]
                    t1s = sink.records[line, k, 1]
                    t2s = sink.records[line, k, 2]
                    block = wp.mat33(
                        sink.diag[line, slot + 1], t12, t1s,
                        t12, sink.diag[line, slot + 2], t2s,
                        t1s, t2s, sink.diag[line, slot + 3],
                    )  # fmt: skip
                    values, vectors = friction_ball_metric(
                        block, rows.row_mu[line, slot + 1], rows.row_mu[line, slot + 3]
                    )
                    basis = wp.quat_from_matrix(vectors)
                    sink.records[line, k, 0] = values[0]
                    sink.records[line, k, 1] = values[1]
                    sink.records[line, k, 2] = values[2]
                    sink.records[line, k, 3] = 0.0
                    sink.records[line, k, 4] = basis[0]
                    sink.records[line, k, 5] = basis[1]
                    sink.records[line, k, 6] = basis[2]
                    sink.records[line, k, 7] = basis[3]
            for r in range(unit.rows):
                incident = sink.row_incident[line, slot + r]
                row_bias(rows, bias, line, slot + r, incident)
            item += threads

    name = f"support_rows_{len(node)}_{abs(hash((parent, first, node))) % (1 << 32):08x}"
    name += ("_records" if records else "") + ("_torsion" if torsion else "")
    # The writer's line, at which no walker spills.
    return python_kernel(kernel, name, launch_bounds=WRITERS["support"].bounds)


def support_threads(units: int, device, writer: WriterShape = WRITERS["support"]) -> int:
    """A striding writer's lanes (:func:`support_rows_kernel`; :func:`write_free_body_rows` at its own ``writer``
    shape): two waves of the blocks a multiprocessor its line keeps, at most ``units`` (the list's or the contacts'
    bound); one thread on the CPU."""
    device = wp.get_device(device)
    if not device.is_cuda:
        return 1
    block, blocks = writer.bounds
    return max(min(2 * blocks * block * int(device.sm_count), int(units)), 1)


# ---------------------------------------------------------------------------------------------------------------------
# The sparse diagonal (case K2): a tree beside a star, its rows into the slots sink (FL3.5)
# ---------------------------------------------------------------------------------------------------------------------
#
# A world of the sparse diagonal pairs a dense arm (its tree walks) with a star of one-coordinate keys on a fixed root.
# A row's seed on the arm walks into the row's dense part, the arm's coordinates in the walk's order; a seed on a key is
# one step to the world, z = (s . w) / ell, the key's step J entry over the root of its pivot (the J / sqrt(m) slot,
# 1 / ell = sqrt(m^-1) of the factor's diagonal), into the row's two slots. The sweep reads the dense part and the slots
# (:func:`.sweep_routes_diag.slot_rows`) and leaves the arm by the walk (:func:`.factor.walk.walk_leave_kernel`), the
# release its rows carry.

SLOT_ROWS = (
    Field("single_dof", wp.int32, ("body",), owner="forest", attr="body_single_response_dof"),
    Field("dof_offset", wp.int32, ("articulation",), owner="forest", attr="articulation_world_dof_offset"),
    Field("inverse_mass", wp.float32, ("dof",), owner="factor", attr="diagonal_inverse_mass"),
    Field("star", wp.int32, attr="diagonal_size", owner="classes"),
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
        classes=solver.spec.classes,
        Z=placement.J_by_size[solver.spec.classes.dense_size],
    )


@wp.func
def key_step(slots: SlotRows, v_hat: wp.array[float], dof: int, j: float):
    """A row's ``J`` entry ``j`` on coordinate ``dof`` of a key of the star, one step to the world: its footprint ``z =
    j / ell`` (``1 / ell = sqrt(m^-1)``) and its incident velocity ``j v_hat``, in the slot of the key's world
    coordinate. A contact's seed (:func:`star_step`) and a joint row's covector (:func:`slot_header`) both take it."""
    return j * wp.sqrt(slots.inverse_mass[dof]), j * v_hat[dof]


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
    """A contact's seed on the star: :func:`key_step` of the step's ``J`` entry of ``sign``'s contact row ``row`` along
    ``direction`` at ``point`` (:func:`.contact_laws.contact_row_entry`); ``(-1, 0, 0)`` for a body without a single
    response coordinate in its articulation's window."""
    coordinate = int(-1)
    value = float(0.0)
    incident = float(0.0)
    dof = slots.single_dof[body]
    if dof >= 0:
        local = dof - data.dof_start[art]
        if local >= 0 and local < slots.star:
            coordinate = slots.dof_offset[art] + local
            j = contact_row_entry(row, data.motion[dof], direction, point, data.origin[art]) * sign
            value, incident = key_step(slots, v_hat, dof, j)
    return coordinate, value, incident


@cache
def slot_header(keys: bool) -> wp.Function:
    """Finish the sparse diagonal's row ``unit.slot`` once the walk wrote its arm's footprints, norm and incident
    (:func:`packed_walker` with the star): each endpoint on a key of the star, one step to the world into the row's two
    slots (:func:`star_step`, merged into the first on one coordinate), its diagonal ``(|z|^2 + sum_s z_s^2) + cfm``, its
    incident velocity with the keys' and its rhs by the row law (:func:`.kernels.row_bias`). With ``keys`` a joint
    row's covector ``value0 e_dof0`` on a key (a velocity limit's: the star's position limits are fused) is one step
    to the world too (:func:`key_step`), into the first slot."""

    def header(
        rows: DenseRows,
        slots: SlotRows,
        data: ContactInput,
        v_hat: wp.array[float],
        unit: RowUnit,
        bias: RowBias,
    ):
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
                    coord_a, z_a, key = star_step(
                        slots, data, v_hat, unit.art_a, unit.body_a, 1.0, row, direction, point
                    )
                    incident += key
            if unit.body_b >= 0 and unit.art_b >= 0:
                if data.response_dofs[unit.art_b] == slots.star:
                    coord_b, z_b, key = star_step(
                        slots, data, v_hat, unit.art_b, unit.body_b, -1.0, row, direction, point
                    )
                    incident += key
        if wp.static(keys):
            if unit.dof0 >= 0 and unit.art_a >= 0:
                if data.response_dofs[unit.art_a] == slots.star:
                    coord_a = slots.dof_offset[unit.art_a] + (unit.dof0 - data.dof_start[unit.art_a])
                    z_a, key = key_step(slots, v_hat, unit.dof0, unit.value0)
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
        row_bias(rows, bias, line, slot, incident)

    header.__name__ = header.__qualname__ = "slot_header"
    return wp.func(header)


read_row_unit = unit_reader(False)
"""The unit reader of a row a unit, without link rows (:func:`unit_reader`): the sparse diagonal's and the free
bodies'."""


@cache
def slot_rows_kernel(side: bool, tables: tuple, keys: bool = False) -> wp.Kernel:
    """The one writer on the sparse diagonal: a world's ``lanes`` threads stride over its rows (``dim = (worlds,
    lanes)``), every row up to the allocation's watermark (``counts``) within the capacity, a row a lane: a contact
    row's records by the contact law, then each row as its own unit (:data:`read_row_unit`), walked by the one unit
    walker with the star (:func:`packed_walker`; the spin row a torque) and finished (:func:`slot_header`). ``side``
    and ``tables`` as :func:`packed_walker`'s: the walk of the solver's classes (:func:`slot_writer`); ``keys``: a
    joint row can sit on a key (:func:`slot_header`)."""
    walk = packed_walker(1, False, side, tables, star=True)
    header = slot_header(keys)

    def write_slot_rows(
        rows: DenseRows,
        support: RowSupport,
        slots: SlotRows,
        sink: RowSink,
        counts: wp.array[int],
        lanes: int,
        data: ContactInput,
        law: PatchRowLaw,
        v_hat: wp.array[float],
        bias: RowBias,
    ):
        world, lane = wp.tid()
        end = wp.min(counts[world], support.units.shape[1])
        line = rows.first[world]
        for slot in range(lane, end, lanes):
            unit = read_row_unit(rows, data, law, JointSeeds(), line, slot, support.units[line, slot])
            if unit.rows > 0:
                unit.group = slots.dense_group[world]
                walk(sink, unit)
                header(rows, slots, data, v_hat, unit, bias)

    name = "write_slot_rows" + ("_keys" if keys else "")
    return python_kernel(write_slot_rows, name, launch_bounds=WRITERS["slots"].bounds)


def slot_writer(solver) -> wp.Kernel:
    """The sparse diagonal's one writer for the solver's walk (:func:`slot_rows_kernel`): the records' side where a
    walked joint can have two halves (:attr:`.tree_factor.TreeFactor.walk_side`) and the walking classes' paths
    (:attr:`.tree_factor.TreeFactor.walk_classes`); a key's joint rows where the step writes velocity-limit rows."""
    factor = solver.factor
    keys = bool(solver.options.enable_joint_velocity_limits)
    return slot_rows_kernel(factor.walk_side, walk_tables(factor.walk_classes), keys)


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
    row (:class:`.row_records.RowSupport`) and the entries, the general placement's packed rows, or the sparse diagonal's
    dense parts by size group (:data:`SLOT_ROWS`' ``Z``) with its star's coordinates."""
    placement = solver.placement
    sink = row_sink(solver.factor, solver.forest, placement, solver.v_hat)
    sink.layout = placement.row_support()
    if solver.spec.classes.case == "K2":
        sink.entries = placement.J_by_size[solver.spec.classes.dense_size]
        sink.star = int(solver.spec.classes.diagonal_size)
    else:
        sink.entries = placement.packed_Z
    return sink


# ---------------------------------------------------------------------------------------------------------------------
# The free bodies' rows: packed rows of the line, each side six entries
# ---------------------------------------------------------------------------------------------------------------------

walk_body_row = packed_walker(1, False, True, walk_tables((FREE_BODY_WALK,)))
"""The one unit walker into the free bodies' rows (:func:`packed_walker`), packed rows of the line in the sides layout:
a row's sides in its ``a``/``b`` order, six entries each."""


@wp.func
def free_body_contact_rows(sink: RowSink, rows: DenseRows, data: ContactInput, law: PatchRowLaw, bias: RowBias, c: int):
    """Contact ``c``'s rows where the allocation routed it to the free bodies (``path`` 1): the contact's front end
    formed once (:func:`.contact_input.contact_frame`), then each row's records by the contact law and the row as its
    own unit (:func:`contact_row_unit`), walked by the one unit walker into the free bodies' rows
    (:data:`walk_body_row`; the spin row a torque), with its diagonal ``|z|^2 + cfm``, its incident velocity and its rhs
    by the row law (:func:`.row_kernels.row_bias`), as every writer's."""
    slot = data.slot[c]
    if data.path[c] != 1 or slot < 0:
        return
    line = sink.layout.first[data.world[c]]
    # The allocation's extent: recomputing eligibility can cross a floating-point threshold.
    count = int(1)
    if data.slots_needed[c] >= 3:
        count = data.slots_needed[c]
    frame = contact_frame(data, law, c)
    for row in range(count):
        walk_body_row(sink, contact_row_unit(data, law, frame, c, row, line, slot + row))
        row_bias(rows, bias, line, slot + row, sink.row_incident[line, slot + row])


@wp.func
def speed_axis(sink: RowSink, line: int, slot: int, art: int, dof: int) -> float:
    """A speed row's epilogue, after its seed's walk gave its diagonal ``|L^-1 P e_k|^2 + cfm``: its ``Z`` holds the
    axis it bounds unwhitened (``e_k`` at the entry of its side's coordinate ``k``, every other entry zero), and its
    incident velocity is zero (returned): the speed law reads the body's velocity through its factor, not a row, from
    the row's three coordinates (:func:`.sweep.speed_row`), and its effective mass converts the speed correction into
    the impulse the row stores. Every writer of a speed row's (:data:`.row_records.UNIT_SPEED`)."""
    layout = sink.layout
    k = dof - sink.walk.dof_start[art]
    entry = k
    if layout.dense != 0:
        entry = layout.offset[art] + k
    for e in range(sink.entries.shape[2]):
        sink.entries[line, slot, e] = 0.0
    sink.entries[line, slot, entry] = 1.0
    sink.row_incident[line, slot] = 0.0
    return 0.0


@wp.func
def free_body_speed_rows(
    bodies: FreeBodies, rows: DenseRows, sink: RowSink, speed_slot: wp.array[int], bias: RowBias, candidate: int
):
    """Free rigid body ``candidate``'s speed rows in the free bodies' run, in the slots and with the records and unit
    sources their allocation wrote (:func:`.row_kernels.allocate_rigid_velocity_limit_slots`): each the seed of the
    axis it bounds at the body's root, walked (:data:`read_row_unit`, :data:`walk_body_row`), then its axis
    (:func:`speed_axis`) and its rhs by the row law (:func:`.row_kernels.row_bias`), as the one writer's."""
    body = bodies.free_rigid_body_indices[candidate]
    art = bodies.body_to_articulation[body]
    if art < 0:
        return
    world = bodies.art_to_world[art]
    if world < 0:
        return
    line = rows.first[world]
    for k in range(RIGID_SPEED_ROWS):
        slot = speed_slot[RIGID_SPEED_ROWS * candidate + k]
        if slot < 0:
            continue
        row = sink.base + slot
        source = unit_source(UNIT_SPEED, 2 * (sink.walk.dof_start[art] + 3 * k))
        unit = read_row_unit(rows, ContactInput(), PatchRowLaw(), JointSeeds(), line, row, source)
        walk_body_row(sink, unit)
        row_bias(rows, bias, line, row, speed_axis(sink, line, row, art, unit.dof0))


@wp.kernel(enable_backward=False, launch_bounds=WRITERS["free_body"].bounds)
def write_free_body_rows(
    sink: RowSink,
    rows: DenseRows,
    data: ContactInput,
    law: PatchRowLaw,
    bodies: FreeBodies,
    speed_slot: wp.array[int],
    speed_bodies: int,
    threads: int,
    bias: RowBias,
):
    """The free bodies' rows (the line's packed rows from ``body_base``), once their families' slots are reserved:
    ``threads`` lanes stride over their units, the step's contacts (:func:`free_body_contact_rows`, those the
    allocation routed to the free bodies), then the ``speed_bodies`` free rigid bodies' speed rows
    (:func:`free_body_speed_rows`). Every unit's seeds walk into the free-body sink with the row law after the walk; a
    unit's rows are its own, so the stride changes no row."""
    contacts = data.count[0]
    for item in range(wp.tid(), contacts + speed_bodies, threads):
        if item < contacts:
            free_body_contact_rows(sink, rows, data, law, bias, item)
        else:
            free_body_speed_rows(bodies, rows, sink, speed_slot, bias, item - contacts)


def free_body_sink(solver) -> RowSink:
    """The free bodies' sink (:class:`RowSink`, :data:`walk_body_row`): the packed rows of the line from the store's
    ``body_base`` (``base``: the free bodies' slots count from it), each side's entries after the row's first side (the
    packed rows' sides layout, whatever the dense rows' layout: :func:`.sweep_routes.free_body_rows`)."""
    placement = solver.placement
    sink = row_sink(solver.factor, solver.forest, placement, solver.v_hat)
    layout = placement.row_support()
    layout.dense = 0
    sink.layout, sink.entries, sink.base = layout, placement.packed_Z, int(placement.body_base)
    return sink

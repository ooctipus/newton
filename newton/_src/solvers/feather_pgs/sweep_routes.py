# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""FeatherPGS's general (matrix-free) sweep as a configuration of the sweep emitter (:mod:`.sweep`).

:class:`GeneralSweep` is its configuration, which the one factory builds (:func:`.sweep.sweep_kernel`); its working
set fits the stage budget of the device it is built for (:func:`.memory.arch_budget`). The module also holds what
every claim's sweep shares: the world queue's dispatch (:func:`worlds`), the free bodies' rows and walk.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from textwrap import dedent

import warp as wp

from ...core.execution import Work, device_lanes
from ...core.memory import REGISTERS_PER_THREAD, Buffer, Operand, Place, Staging
from . import kernel_template, sweep_units
from .device_preludes import LANE_WORDS, memory_words
from .factor.walk import WalkTrees
from .kernel_template import Dispatch
from .packed_rows import CLASS_SIDES, PACKED_SIDES_BY_SIDE, PACKED_SIDES_DENSE
from .projection_laws import IDLE_LAW
from .region_laws import REGION_LAW, REGION_SLIDE
from .region_sweep import (
    FRICTION_ROWS,
    KIND_BLOCK,
    KIND_SECOND,
    KIND_SPRINGS,
    PAIR_BLOCKS,
    REGION_IDLE,
    REGION_LEFT,
    REGION_LOOSE,
    WALK_PARTS,
    walk_levels,
)
from .response_operator import Lanes, SparseFactor, Velocity
from .row_inputs import RowArgument
from .row_laws import ROW_CONTACT, ROW_FRICTION, ROW_JOINT_LIMIT, ROW_TORSION
from .row_metadata import record_words
from .row_records import line_size
from .solve_policy import QUIET_LITERAL, block_argmax, sweep_exit
from .sweep import (
    EXACT,
    FORMED_RECORD,
    FREE_HEAD,
    FRICTION_METRIC_FLOATS,
    ITERATION,
    SEQUENTIAL,
    Diagonal,
    Levels,
    RowContext,
    RowLaws,
    SweepConfig,
    UnitZ,
    WorldRows,
    active_scan,
    active_selection,
    active_violation,
    active_words,
    cached_pair_law,
    contact_unit,
    end_sweep,
    formed_friction_laws,
    friction_metric,
    friction_record_stage,
    impulse_seed,
    indent,
    level_loop,
    row_step,
    rows_block,
    triple_phase,
    unit_visit,
)
from .sweep_levels import COST_MATES


def worlds(
    groups: int | None = None, lanes: int | None = None, classes: bool = False, ranks: str | None = None
) -> kernel_template.Dispatch:
    """The worlds a sweep launch covers: its claim's bucket of the claims' worlds (:func:`.sweep_levels.claim_worlds`),
    a device list (``general_worlds``, its count first) walked with the grid's stride (``groups`` worlds a block); the
    general sweep's (``classes``) its cost classes, heaviest first, read as one list (``general_claims``, by each
    world's lanes together). Where lane groups of ``lanes`` share a warp and the classes hold heavy worlds
    (:data:`.sweep_levels.COST_MATES` classes above the middle world's), each warp's first world is the list's next
    from its head and its others the next from its tail (:func:`~newton._src.core.execution.warp_mates`): a heavy world
    starts first with the lightest mates. A list of like costs is walked in order, neighbours (of their own cost, near
    their data) sharing a warp. With ``ranks`` (a launch operand), a world takes that many consecutive blocks: its
    first (rank 0) runs the world, the others serve its levels wider than a block (:func:`region_phase`)."""
    if not classes:
        return kernel_template.items(
            "world",
            groups=groups,
            lanes=lanes,
            queue=("general_world_count", "general_world_grid_stride", "general_worlds"),
            ranks=ranks,
        )
    mates = 32 // lanes if groups is not None and lanes is not None and lanes < 32 else 1
    return kernel_template.items(
        "world",
        groups=groups,
        lanes=lanes,
        span=("general_claims", "general_world_grid_stride"),
        mates=min(mates, groups or 1),
        above=COST_MATES,
    )


MF_GS_ABI = kernel_template.params(
    ("world", int),
    ("world_constraint_count", wp.array[int]),
    ("dense_first", wp.array[int]),
    ("topology_rows", wp.array[int]),
    ("world_dof_indices", wp.array2d[int]),
    ("rhs_bias", wp.array2d[float]),
    ("row_incident", wp.array2d[float]),
    ("world_diag", wp.array2d[float]),
    ("world_row_w", wp.array2d[float]),
    ("world_impulses", wp.array2d[float]),
    ("body_base", wp.array[int]),
    ("response_dofs", wp.array[int]),
    ("articulation_world_dof_offset", wp.array[int]),
    ("packed_dense", int),
    ("packed_Z", wp.array3d[float]),
    ("row_sides", wp.array2d[wp.vec2i]),
    ("world_row_type", wp.array2d[int]),
    ("world_row_parent", wp.array2d[int]),
    ("world_row_mu", wp.array2d[float]),
    ("mf_constraint_count", wp.array[int]),
    ("mf_contact_rows_end", wp.array[int]),
    ("row_record", wp.array2d[int]),
    ("walk", WalkTrees),
    ("iterations", int),
    ("omega", float),
)
"""The matrix-free sweep's native ABI: one world per call (the world queue's dispatch). The free bodies' run's rows are
the row store's (the ``world_*`` records) and the packed rows' (``packed_Z``), from the dense capacity of the world's
line; the streamed rows' records the line's (``row_record``, :func:`.row_metadata.pack_row_record`)."""


_ROW_BUFFERS = {"metadata": "s_rec_dense", "impulses": "s_lam_mf", "dense_impulses": "s_lam_dense"}
"""Each row operand's buffer in the block's stage (:func:`mf_gs_staging`)."""

FREE_IMPULSES = _ROW_BUFFERS["impulses"]
"""The free bodies' staged impulses, the buffer that keeps a head where a block cannot hold every row's."""


@dataclasses.dataclass(frozen=True)
class RowPlaces:
    """Where the matrix-free sweep's row operands live, each in the block's stage or at the alternative place the stage
    gives it where it declines it (:func:`.memory.stage`), with identical values.

    Args:
        metadata: The dense rows' metadata (with their sides): staged, or their records streamed.
        impulses: The free bodies' impulses: staged, or in place.
        dense_impulses: The dense rows' impulses (the line's): staged, or in place.
        velocity: The world's velocity change (``s_v``, and ``s_sweep_v`` where rows are wide): staged, or a line of
            global memory a world (``sweep_velocity``) where a block cannot hold it beside every other operand at its
            alternative place (a world past a block's reach; :meth:`.sweep_plan.Sweep.build`).
        free_slots: Where the free bodies' impulses are staged but a block cannot hold every row's, the rows staged
            (fixed slots, the stage's head: :attr:`.memory.Staging.heads`): a world whose live rows fit them stages
            them, a world past them keeps every impulse in place (the spill). ``0``: every row's at ``impulses``.
        wide_slots: Where the block's budget holds a longer head than the stage that keeps the L1
            (:attr:`.memory.Staging.wide`), its slots: the sweep launches twice, at each head's shared memory, and a
            world is swept by the launch whose head holds its live rows (past both, the first's, in place), so its
            stage follows its live rows, not the capacity. ``0``: one head (``free_slots``) or none.
        dense_rows: The dense rows this launch's stage holds, a head of the capacity (their metadata at its word
            stride, their impulses, the active enumeration's bitsets and the levels' marks): the launch sweeps the
            worlds whose live dense rows pass ``dense_floor`` and fit it (:meth:`.sweep_plan.Sweep._dense_heads`).
            ``0``: the capacity, every world past ``dense_floor``.
        dense_floor: The dense rows of the worlds an earlier launch's head holds: this launch takes the worlds past
            them. ``0``: from the first.
    """

    metadata: Place = Place.SHARED
    impulses: Place = Place.SHARED
    dense_impulses: Place = Place.SHARED
    velocity: Place = Place.SHARED
    free_slots: int = 0
    wide_slots: int = 0
    dense_rows: int = 0
    dense_floor: int = 0

    def __post_init__(self):
        if (self.free_slots or self.wide_slots) and self.impulses is not Place.SHARED:
            raise ValueError("the free bodies' impulses spill past their staged slots: they are staged")
        if self.wide_slots and self.wide_slots <= self.free_slots:
            raise ValueError("a wide head holds more slots than the stage's")
        if self.dense_rows and self.dense_rows <= self.dense_floor:
            raise ValueError("a dense head holds more rows than the earlier heads")

    def __repr__(self) -> str:
        dense = f", {self.dense_impulses.name}" if self.dense_impulses is not Place.SHARED else ""
        velocity = f", velocity {self.velocity.name}" if self.velocity is not Place.SHARED else ""
        slots = f", {self.free_slots} slots" if self.spilled else ""
        wide = f" or {self.wide_slots}" if self.wide_slots else ""
        head = f", dense rows {self.dense_floor} to {self.dense_rows or 'all'}" if self.dense_headed else ""
        return f"RowPlaces({self.metadata.name}, {self.impulses.name}{slots}{wide}{dense}{velocity}{head})"

    @property
    def dense_headed(self) -> bool:
        """Whether the launch sweeps only the worlds whose live dense rows its head range holds (:attr:`dense_rows`)."""
        return bool(self.dense_rows or self.dense_floor)

    @property
    def spilled(self) -> bool:
        """Whether the free bodies' impulses keep a head of their rows staged, the rest in place."""
        return bool(self.free_slots or self.wide_slots)

    def placed(self, staging: Staging) -> RowPlaces:
        """These places, each operand the stage held (:meth:`.memory.Staging.place`) at the place it got, and the free
        bodies' impulses' slots where the stage kept their head (and the wide head's, :attr:`.memory.Staging.wide`)."""
        held = {buffer.name for buffer in (*staging.buffers, *staging.declined)}
        moved = {field: staging.place(name) for field, name in _ROW_BUFFERS.items() if name in held}
        impulses = FREE_IMPULSES
        if impulses in staging.heads:
            moved["impulses"] = Place.SHARED
            moved["free_slots"] = next((buffer.count for buffer in staging.buffers if buffer.name == impulses), 0)
            moved["wide_slots"] = staging.wide
        return dataclasses.replace(self, **moved)

    @property
    def suffix(self) -> str:
        """The places' part of a kernel's name."""
        return (
            ("" if self.metadata is Place.SHARED else "_gmeta")
            + ("" if self.impulses is Place.SHARED else "_glambda")
            + (f"_slots{self.free_slots}" if self.spilled else "")
            + (f"to{self.wide_slots}" if self.wide_slots else "")
            + ("" if self.dense_impulses is Place.SHARED else "_gline")
            + ("" if self.velocity is Place.SHARED else "_gv")
            + (f"_d{self.dense_floor}to{self.dense_rows or 'all'}" if self.dense_headed else "")
        )


STAGED_ROWS = RowPlaces()
"""Every row operand in the block's shared stage, the default."""


def velocity_fits(config, budget: int) -> bool:
    """Whether a block holds ``config``'s velocity change in its stage beside every operand that has no alternative
    place (each other at its alternative): where it does not, the world is past a block's reach."""
    staging = config.staging()
    return staging.only({buffer.name for buffer in staging if buffer.alternative is None}).fits(budget)


def record_first(max_constraints: int, places: RowPlaces) -> int:
    """The line's first row with a record (:func:`.row_metadata.pack_row_record`): the dense rows' first where their
    metadata streams, else the free bodies' (from the dense capacity ``max_constraints``), which always stream."""
    return 0 if places.metadata is Place.STREAMED else int(max_constraints)


ROW_PLACES = (
    STAGED_ROWS,
    RowPlaces(impulses=Place.GLOBAL),
    RowPlaces(Place.STREAMED),
    RowPlaces(Place.STREAMED, Place.GLOBAL),
    RowPlaces(Place.STREAMED, Place.GLOBAL, Place.GLOBAL),
)
"""Every placement of the sweep's row operands (the census's axis)."""


RUN_AHEAD_WORDS = (("i + 1", 1), ("tangent", 1), ("i + 2", 2), ("sibling", 2), ("(i) + 3", 3))
"""The kind and parent words a grouped run's visit reads past its own row (:attr:`.sweep.WorldRows.ahead`): a contact
unit's test of its pair (``i + 1``, ``i + 2``, the unit's ``tangent`` and ``sibling``) and of a spin row (``i + 3``)."""

RUN_KEPT_STORES = 4
"""The stores a grouped run keeps for its patch rings' walks (:attr:`.sweep.WorldRows.kept_stores`): a patch's
normals are visited in turn just before its friction pair, so its ring's members are the run's last stores; a longer
ring's others load. A hardware choice that changes no result: a member read there is the value its load reads."""

RUN_AHEAD_ENTRIES = (("i", 0), ("(parent + 2)", 1), ("spin", 2), ("tangent", 1), ("sibling", 2))
"""The rows whose entries a grouped run's visit reads (:attr:`.response_operator.SparseFactor.ahead`): the visited
row, a first tangent's sibling and spin row (rows ``i + 1`` and ``i + 2`` at its visit), a unit's pair."""


def packed_operator(
    support: int,
    lanes: Lanes,
    *,
    wide: bool = False,
    rows: WorldRows | None = None,
    classes: bool = False,
    held: bool = False,
) -> SparseFactor:
    """The dense rows of a matrix-free sweep: packed whitened rows of ``support`` entries (:mod:`.packed_rows`), their
    world coordinates from their sides (``PACKED_COORDINATE``; where ``rows`` stream their records, from the sides
    their records hold, ``RECORD_COORDINATE``; with ``classes``, by their support classes too, ``CLASS_*``), or with
    ``wide`` rows every coordinate of the world, and values from the world's first row (``packed_base``), over the
    factor velocity change the sweep stages (``s_v``; ``held`` where it lies in global memory, :attr:`Velocity.held`)."""

    def coordinate(row, entry):
        if wide:
            return f"(world_dof_indices.data[dof_map_base + ({entry})] >= 0 ? ({entry}) : -1)"
        if rows is not None and (rows.record or rows.sides):
            split = rows.sides_of(row)
            if split is not None:
                # A split record's sides take a word each (row_metadata.split_records).
                if classes:
                    return f"CLASS_ENTRY({split[0]}, {split[1]}, {rows.word(row, 4)}, {entry})"
                widths = rows.word(row, 4)
                return (
                    f"SIDES_COORDINATE({split[0]}, {widths} & 0xFFFF, {split[1]}, ({widths} >> 16) & 0xFFFF, {entry})"
                )
            prefix = "CLASS_RECORD" if classes else "RECORD"
            return f"{prefix}_COORDINATE({rows.word(row, 0)}, {rows.word(row, 4)}, {entry})"
        if classes:
            return (
                f"CLASS_COORDINATE(row_sides.data[off_dense + ({row})], row_classes.data[off_dense + ({row})], {entry})"
            )
        return f"PACKED_COORDINATE(row_sides.data[off_dense + ({row})], {entry})"

    def value(row, entry):
        return f"packed_Z.data[packed_base + ({row}) * {int(support)} + ({entry})]"

    def key(row):
        # The words a record's coordinates decode from (``coordinate``): its sides' and its classes' or widths'.
        split = rows.sides_of(row)
        return (*(split if split is not None else (rows.word(row, 0),)), rows.word(row, 4))

    keyed = rows is not None and rows.record and not wide
    return SparseFactor(
        int(support), coordinate, value, lanes, velocity=Velocity("s_v", held=held), key=key if keyed else None
    )


def free_body_rows(lanes: Lanes, support: int | str = 12) -> SparseFactor:
    """The free bodies' rows, whitened: ``Z = L^-1 J^T`` over at most two free bodies, the 6 + 6 support class.

    A free body is a one-node tree; ``L`` is its held factor (``H = L L^T``, armature included), and the sweep holds the
    change of the body's six velocity coordinates in factor coordinates, ``u = L^T dv`` (``s_v``). Its rows are packed
    rows of the line (:mod:`.packed_rows`, ``support`` entries a row) from the free bodies' first row's ``free_base``,
    in the sides layout: entry ``e < 6`` at body ``a``'s coordinate ``dof_a + e``, entry ``e >= 6`` at body ``b``'s
    ``dof_b + e - 6``, none for a negative body (the visit's ``dof_a``, ``dof_b``, which every row of a contact shares).
    Twelve entries hold a row, or ``support`` where the line is narrower (no world holds two free bodies, so no row
    has a second side; a ``support`` known only at run time, a native expression, reads twelve, each where its
    coordinate exists). The builder stores ``1 / A_ii`` (``A_ii = cfm + |Z|^2``) in the row's header word.

    ``Z``'s place: with lanes, lane ``l < 12`` owns entry ``l`` and the visited row's (``row``) is the free bodies' row
    loop's register ``cur_z``, loaded a row ahead (:func:`.sweep.rows_block`); every other row, its friction sibling
    (``sib``), spin row (``spin``) or a seed's row, is read where it is used. One thread reads every row where it is
    used, the visited one at the loop's ``i``. The loop's rows do not own their stores, so a friction pair applies its
    second tangent on its own (``fused_pair_apply`` false).
    """

    def coordinate(row: str, entry: str) -> str:
        # The side's body first, then its coordinate: a guard the compiler folds into the body's test.
        return f"(({entry} < 6 ? dof_a : dof_b) < 0 ? -1 : ({entry} < 6 ? dof_a + {entry} : dof_b + {entry} - 6))"

    def value(row: str, entry: str) -> str:
        if row == "row":
            if lanes.parallel:
                return "cur_z"
            row = "i"
        return f"packed_Z.data[free_base + {row} * {support} + {entry}]"

    return SparseFactor(
        12 if isinstance(support, str) else min(12, int(support)),
        coordinate,
        value,
        lanes,
        velocity=Velocity("s_v"),
        registers=True,
        fused_pair_apply=False,
    )


BLOCK_WARPS, WARP_LINE = 2, 64
"""The warp mapping's block where its worlds' stages fit, and its register line then (the sparse sweeps')."""

COLORED_THREADS = 256
"""Threads of a colored world's block: a thread takes every ``COLORED_THREADS``-th unit in color order. The block is the
world's (its staging, its barriers and its ``tid``), so the mapping declares it; the layer would otherwise pack worlds
of 32 lanes or fewer into one block."""

COLOR_SLOT = "LOAD_slots(reinterpret_cast<const int4*>(&color_slot.data[(off_unit + {}) * 4]))"
"""The colored unit at a place of its world's colors: its first row, rows and bodies (:data:`.sweep_units.UNIT_FIELDS`)."""

RAN_SLOT = 6
"""A colored unit's body's slot in ``s_ran``: its first coordinate over its six. Both sides of a colored unit are free
bodies of six coordinates each, so two bodies' first coordinates lie six apart at least and their slots differ."""


def ran_slots(dofs: int) -> int:
    """Slots of ``s_ran`` for a world of ``dofs`` coordinates (:data:`RAN_SLOT`)."""
    return int(dofs) // RAN_SLOT + 1


# A world's group of blocks (worlds' ranks): the block's rank among them and their count follow the world.
GROUP_ABI = kernel_template.params(MF_GS_ABI[0], ("rank", int), ("group", int), *MF_GS_ABI[1:])

COLOR_AFTER = "LOAD_slots(reinterpret_cast<const int2*>(&color_after.data[(off_unit + {}) * 2]))"
"""The colored unit's turn at a place: the colors of the latest earlier unit on each of its bodies
(:data:`.sweep_units.UNIT_FIELDS`)."""

COLORED_ABI = kernel_template.params(
    *MF_GS_ABI,
    ("world_slot", wp.array[int]),
    ("slot_first", wp.array[int]),
    ("slot_need", wp.array[int]),
    ("color_count", wp.array[int]),
    ("color_offset", wp.array[int]),
    ("color_slot", wp.array2d[int]),
    ("color_after", wp.array2d[int]),
    ("slot_data", wp.array2d[float]),
    ("slot_words", wp.array2d[int]),
    ("v_out", wp.array[float]),
)
"""The colored mapping's native ABI: the matrix-free sweep's, then each world's storage slot and its slot's first entry
and entries of the unit pool, the worlds' color counts, and the pool's colors, staged units and friction records
(:data:`.sweep_units.WORLD_FIELDS`, :data:`.sweep_units.SLOT_PLACES`, :data:`.sweep_units.UNIT_FIELDS`,
:data:`.sweep_units.SLOT_FIELDS`) before ``v_out``."""

COLORED_GROUP_ABI = kernel_template.params(COLORED_ABI[0], ("rank", int), ("group", int), *COLORED_ABI[1:])
"""The colored mapping's ABI over a world's group of blocks (its records stage, :attr:`GeneralSweep.spread`)."""


_UNIT_ROWS = (("n", "unit_jv0", "unit_dn"), ("t1", "unit_jv1", "unit_dt1"), ("t2", "unit_jv2", "unit_dt2"))
"""A colored unit's rows: each one's partial sum, velocity term and deferred change, at ``E``'s ``12 r``."""
_SPIN_ROW = ("spin", "unit_jv3", "unit_ds")
"""Under contact torsion the unit's fourth row, its spin row."""


def _unit_rows(torsion: bool) -> tuple:
    return (*_UNIT_ROWS, _SPIN_ROW) if torsion else _UNIT_ROWS


def _unit_terms(pad: str, *, torsion: bool = False) -> str:
    """Native statements, after a unit's ``E``, ``rows``, ``dof_a``, ``dof_b`` and its record (its couplings in
    ``friction_m3``) are declared: the unit's velocity at its staged rows' entries (``velocities``,
    :func:`_staged_rows`), its rows' velocity terms from it (each side's six products, then the sides added) and the
    normal's couplings with its pair (and under ``torsion`` its spin row, whose term it reduces too)."""
    rows = _unit_rows(torsion)
    couplings = zip(("gram_n1", "gram_n2", "gram_n3")[: len(rows) - 1], "yzw", strict=False)
    lines = [
        "float velocities[12];",
        "#pragma unroll",
        "for (int k = 0; k < 6; ++k) {",
        "    velocities[2 * k] = dof_a >= 0 ? s_v[dof_a + k] : 0.0f;",
        "    velocities[2 * k + 1] = dof_b >= 0 ? s_v[dof_b + k] : 0.0f;",
        "}",
        f"float {', '.join(f'{term} = 0.0f' for _, term, _ in rows)};",
        "#pragma unroll",
        "for (int side = 0; side < 2; ++side) {",
        "    if ((side == 0 ? dof_a : dof_b) < 0) continue;",
        f"    float {', '.join(f'{part} = 0.0f' for part, _, _ in rows)};",
        "#pragma unroll",
        "    for (int k = 0; k < 6; ++k) {",
        "        const float u = velocities[2 * k + side];",
        *(
            f"        {part} += E[{f'{12 * r} + ' if r else ''}side * 6 + k] * u;"
            for r, (part, _, _) in enumerate(rows)
        ),
        "    }",
        *(f"    {term} += {part};" for part, term, _ in rows),
        "}",
        f"const float {', '.join(f'{name} = {FORMED_RECORD[3]}.{c}' for name, c in couplings)};",
    ]
    return indent("\n".join(lines), pad)


def _staged_rows(torsion: bool) -> SparseFactor:
    """A colored unit's rows as its slot staged them (``E[12 r + 6 side + k]``, row ``r`` of :func:`_unit_rows`), one
    thread's 6 + 6 support class enumerated a coordinate at a time: entry ``2 k + side`` is body ``side``'s coordinate
    ``k``, so an apply writes each coordinate's two bodies together. The unit's velocity at entry ``q`` is
    ``velocities[q]`` (:func:`_unit_terms`)."""
    order = {part: r for r, (part, _, _) in enumerate(_unit_rows(torsion))}

    def coordinate(row: str, entry: str) -> str:
        body = f"(({entry}) & 1 ? dof_b : dof_a)"
        return f"({body} < 0 ? -1 : {body} + (({entry}) >> 1))"

    def value(row: str, entry: str) -> str:
        return f"E[{12 * order[row]} + 6 * (({entry}) & 1) + (({entry}) >> 1)]"

    return dataclasses.replace(free_body_rows(Lanes(1)), index=coordinate, factor=value)


def _unit_apply(pad: str, *, torsion: bool = False) -> str:
    """The unit's deferred changes in one pass on its staged rows (:func:`_staged_rows`): ``u += Z_n dn + Z_t1 dt1 +
    Z_t2 dt2`` on each of its bodies' coordinates (and ``Z_s ds`` of its spin row under ``torsion``), summed in the
    rows' order and added once to the velocity :func:`_unit_terms` read."""
    changes = tuple((part, change) for part, _, change in _unit_rows(torsion))
    lines = [
        f"if ({' || '.join(f'{change} != 0.0f' for _, change in changes)}) {{",
        _staged_rows(torsion).axpy(changes, "    ", kept=True, block=True),
        "}",
    ]
    return indent("\n".join(lines), pad)


# The colored units' slots (sweep_units.SlotLayout): the unit builder writes them a launch before the sweep, and no thread
# of the sweep does, so the memory toolkit's read rule reads them by LDG.
_SLOTS = Operand("slots")


def _colored_units(
    free: WorldRows, laws: RowLaws, M_MF: int, units: str, record: Callable[[str], str], record_stride: str
) -> str:
    """A sweep of the world's units in color order, each when its turn comes. The block takes the places in rounds of
    ``blockDim.x``, a thread a place; a unit loads its fields at once (:data:`.sweep_units.SLOT_FIELDS`, field by field
    across the pool's ``units`` entries) and its contact's ``record`` (:func:`.sweep.friction_record_stage`), then its
    warp polls: each pass runs together the lanes whose turn came, where the latest earlier unit on each of their bodies
    ran (:data:`COLOR_AFTER`: those units' colors, which they published in ``s_ran`` at their bodies' slots,
    :data:`RAN_SLOT`). A unit runs the free-body row law, unrolled over its rows, on the world's staged velocity, and
    publishes its own color on its bodies.

    Every unit thus reads the velocities and impulses that the units before it in row order on its bodies left, as the
    row-order Gauss-Seidel sweep does (its patch ring's rows are its bodies' units too), with no barrier between
    colors: a body's units run one after another in its row order and the others at once, so a sweep takes the
    longest chain of turns rather than a barrier and the slowest unit of every color. No unit waits forever: the
    units a unit waits for sit at earlier places, a warp takes its rounds in order and polls until its lanes ran, so
    the earliest unit not yet run always runs. A sweep's turns follow the last sweep's (``turn``, the sweep's index
    times the colors), so ``s_ran`` needs no clearing between sweeps."""
    unit_rows = WorldRows(
        M_MF, Place.STREAMED, free=True, stream=free.stream, lanes=Lanes(1), slots=free.slots, wide=free.wide
    )
    torsion = laws.torsion
    layout = sweep_units.slot_layout(torsion)

    def unit_offset(row: str) -> str:
        offsets = {"i": "r", "sib": "2", **({"spin": "3"} if torsion else {})}
        if row not in offsets:
            raise ValueError(f"the colored unit reads rows {sorted(offsets)}, not {row!r}")
        return offsets[row]

    ring, members = layout.ring, sweep_units.RING_MEMBERS

    def ring_load(parent: str) -> str:
        # The staged ring's members in ring order (the walk's sum, without its chain of global loads), or the walk.
        words = ", ".join(f"W[{ring + k}]" for k in range(members // 2))
        count = f"W[{ring + members // 2}]"
        return f"({count} < 0 ? {unit_rows.ring_load(parent)} : RING_SUM({unit_rows.load(parent)}, {words}, {count}))"

    unit_ctx = RowContext(
        row="i",
        kind="mf_rt",
        # One thread per unit, every row's velocity term reduced from the unit's velocity ``u`` before the unit's first
        # row (_unit_terms), the normal's couplings and the friction metric from its contact's record, formed once per
        # launch (sweep.friction_record_stage), one pass over the unit's coordinates after its rows (_unit_apply).
        operator=UnitZ(cached=True),
        diagonal=Diagonal(
            inverse=lambda row: "mf_diag" if row == "i" else f"__int_as_float(W[{unit_offset(row)} * 3])"
        ),
        load=unit_rows.load,
        store=unit_rows.store,
        rhs=lambda row: "__int_as_float(meta.z)" if row == "i" else f"__int_as_float(W[{unit_offset(row)} * 3 + 1])",
        mu=lambda row: f"E[{layout.spin_mu if row == 'spin' else layout.mu}]",
        weight=lambda row: f"E[{layout.weight}]",
        eps=unit_rows.eps,
        parent="parent",
        ring_load=ring_load,
        lanes=Lanes(1),
        contact_kinds=(ROW_CONTACT,),
        pair_law="FRICTION_PAIR_CACHED(0, ",
        triple_law="FRICTION_TRIPLE_CACHED(0, ",
    )
    # The friction laws on the unit's record as it loaded it with its rows: the pair metric's (friction_m1), and under
    # torsion the triple's cone law on the ball metric every contact with a spin row has (a zero coefficient pins it).
    laws_text = cached_pair_law(
        "FRICTION_PAIR_CACHED", "friction_m1.x", "friction_m1.y", ("friction_m1.z", "friction_m1.w"), ""
    )
    if torsion:
        laws_text += """
const auto FRICTION_TRIPLE_CACHED = [&](int, const wp::mat33&, wp::vec3 residual, wp::vec3 old, float mu, float mu_t,
                                        float load, float relaxation) -> wp::vec3 {
    return ConeLaw::friction_triple_from_metric(wp::vec3(friction_m0.x, friction_m0.y, friction_m0.z),
        wp::mat33(friction_m1.x, friction_m1.y, friction_m1.z, friction_m1.w, friction_m2.x, friction_m2.y, friction_m2.z,
                  friction_m2.w, friction_m3.x), residual, old, mu, mu_t, load, relaxation);
};"""
    lam = unit_rows.load
    ring_sum = "\n".join(
        [
            "            // A staged patch ring's pooled load: its members' impulses added in ring order.",
            "            const auto RING_SUM = [&](float load, int w0, int w1, int w2, int w3, int members) -> float {",
            "                // Every member's impulse at once (a member past the count reads nothing), then the sum in order.",
            f"                float member[{members}];",
            "#pragma unroll",
            f"                for (int k = 0; k < {members}; ++k) {{",
            "                    const int word = k < 2 ? w0 : k < 4 ? w1 : k < 6 ? w2 : w3;",
            "                    const int r = (int)(((unsigned)word >> ((k & 1) * 16)) & 0xFFFFu);",
            f"                    member[k] = k < members ? {lam('r')} : 0.0f;",
            "                }",
            "#pragma unroll",
            f"                for (int k = 0; k < {members}; ++k)",
            "                    if (k < members) load += member[k];",
            "                return load;",
            "            };",
        ]
    )
    fields, words = layout.fields, layout.words
    declare = f"""const int i = unit_row + r;
int4 meta;
meta.x = 0;
meta.y = W[r * 3];
meta.z = W[r * 3 + 1];
meta.w = W[r * 3 + 2];
const int packed_tp = meta.w;
// The unit builder's rows: a contact, then its friction pair (each copy knows its kind).
const int mf_rt = r == 0 ? {ROW_CONTACT} : {ROW_FRICTION};
const float mf_diag = __int_as_float(meta.y);"""
    unit = contact_unit(
        unit_ctx,
        laws,
        rows="rows",
        declare=declare,
        parent=unit_rows.parent_of("packed_tp"),
        check="mf_diag",
        pad=" " * 20,
    )
    visit = f"""                    const int4 slot = {COLOR_SLOT.format("place")};
                    const int unit_row = slot.x, rows = slot.y, dof_a = slot.z, dof_b = slot.w;
                    const int2 after = {COLOR_AFTER.format("place")};
                    // Every field of the unit at once, beside its slot's own load: one round of memory latency per
                    // unit. A unit without a friction pair never reads its pair's fields. The world's block from its
                    // first entry, field-major over its entries.
                    float E[{fields}];
                    int W[{words}];
#pragma unroll
                    for (int f = 0; f < {fields}; ++f) E[f] = slot_data.data[{fields} * off_unit + f * {units} + place];
#pragma unroll
                    for (int f = 0; f < {words}; ++f) W[f] = slot_words.data[{words} * off_unit + f * {units} + place];
                    // Its contact's record by place, beside the slot (sweep.friction_record_stage): the friction
                    // metric, then the normal's couplings with its friction rows; unread without a friction pair.
                    const float* const unit_record = {record("place")};
                    const float4 {", ".join(f"{name} = make_float4({', '.join('0.0f' if 4 * k + c == 3 else f'unit_record[{4 * k + c} * {record_stride}]' for c in range(4))})" for k, name in enumerate(FORMED_RECORD))};
{indent(laws_text, " " * 20)}"""
    turn_text = f"""                    // Its warp's rounds: each round its units whose turn came run (the latest earlier unit on each
                    // of their bodies ran: their colors in s_ran), together.
                    while (__any_sync(0xffffffffu, pending)) {{
                        if (pending) {{
                            volatile const int* const ran = s_ran;
                            const bool ready = (after.x < 0 || ran[dof_a / {RAN_SLOT}] == turn + after.x)
                                && (after.y < 0 || ran[dof_b / {RAN_SLOT}] == turn + after.y);
                            if (ready) {{
                                __threadfence_block();
{indent(_unit_terms("", torsion=torsion), " " * 32)}
{indent(dedent(unit), " " * 32)}
{indent(_unit_apply("", torsion=torsion), " " * 32)}
                                // Its color on its bodies, after its changes: the next unit on each may run.
                                __threadfence_block();
                                volatile int* const done = s_ran;
                                const int color = turn + max(after.x, after.y) + 1;
                                if (dof_a >= 0) done[dof_a / {RAN_SLOT}] = color;
                                if (dof_b >= 0) done[dof_b / {RAN_SLOT}] = color;
                                pending = false;
                            }}
                        }}
                    }}"""
    return f"""{ring_sum}
            const int colors = color_count.data[world];
            const int units_end = color_offset.data[off_color + colors];
            const int turn = {ITERATION} * colors;
            for (int base = 0; base < units_end; base += blockDim.x) {{
                // A warp's lanes take their places together (a lane past the end loads the last unit and skips it).
                const int place = min(base + tid, units_end - 1);
                bool pending = base + tid < units_end;
{visit}
{turn_text}
            }}
            __syncthreads();"""


def dense_units(
    dense: WorldRows, op: SparseFactor, laws: RowLaws, *, active: bool, record: Callable[[str], str] | None, pad: str
) -> str:
    """The contact unit on the dense rows (:func:`.sweep.unit_visit`), its couplings and pair metric the launch's
    ``record`` (:func:`.sweep.friction_record_stage`) or its own reduction's; a friction triple declined."""
    loads = after = ""
    if record is not None:
        loads = f"""        const float4* const record = reinterpret_cast<const float4*>({record("i")});
        const float4 coupling = record[0], basis = record[1];
"""
        after = "\n        const float gram_n1 = coupling.x, gram_n2 = coupling.y;\n" + cached_pair_law(
            "UNIT_PAIR", "basis.x", "basis.y", ("basis.z", "basis.w"), "        "
        )
    ctx = RowContext(
        row="i",
        kind="row_type",
        operator=UnitZ(cached=record is not None, fused_pair_apply=op.fused_pair_apply),
        diagonal=Diagonal(dense.diag),
        # The unit's rows' impulses as its loop holds them (UNIT_IMPULSE, sweep.rows_block).
        load=lambda row: (
            {"head": "UNIT_IMPULSE(0)", "head + 1": "UNIT_IMPULSE(1)", "head + 2": "UNIT_IMPULSE(2)"}.get(row)
            or dense.load(row)
        ),
        store=dense.store,
        rhs=dense.rhs_of,
        mu=lambda row: "mu",
        weight=dense.weight,
        eps=dense.eps,
        parent="parent",
        ring_load=dense.ring_load,
        lanes=dense.lanes,
        **({"pair_law": "UNIT_PAIR(0, "} if record is not None else {}),
    )
    triple = f" && {dense.spin_of('i', 'm_dense')} < 0" if laws.torsion else ""
    condition = f"{dense.parent('tangent')} == i && {dense.parent('sibling')} == i{triple}"
    return unit_visit(
        dense,
        op,
        laws,
        ctx,
        count="m_dense",
        active=active,
        mu=dense.mu("tangent"),
        guard=("", condition),
        record=(loads, after),
        taken=True,
        pad=pad,
    )


def coordinate_levels(
    dense: WorldRows, op: SparseFactor, laws: RowLaws, *, metric: str | None, formed, triple_phase: bool = False
) -> tuple[str, str, str]:
    """The units on one coordinate no wider unit holds (:func:`.sweep_levels.coordinate_levels_kernel`) by level, a
    lane a unit, before the order, with which they commute: the leveled heads' marks, the phase and the order's skip.
    Under ``triple_phase`` a leveled contact's triple is the triple phase's, after the order, as every triple's."""
    M = dense.capacity
    count, offsets = (
        "sparse_contact_level_count.data[world]",
        f"sparse_contact_level_offsets.data[world * {M + 1} + {{}}]",
    )
    store = dataclasses.replace(dense, lanes=Lanes(1))
    one = dataclasses.replace(op, support=1, index=lambda row, entry: "unit_coordinate", lanes=Lanes(1))

    def visit(row: str, pad: str, **known) -> str:
        step = row_step(store, one, laws, end="m_dense", metric=metric, formed=formed, pad=pad + "    ", **known)
        record = indent(store.visit("i"), pad + "    ") if store.record else ""
        return f"{pad}do {{\n{pad}    const int i = {row};\n{record}{chr(10) if record else ''}{step}\n{pad}}} while (false);"

    pad = " " * 16
    owns = f"head + 2 < m_dense && ({dense.kind('head + 1')}) == {ROW_FRICTION} && {dense.parent('head + 1', enclosed=True)} == head"
    if triple_phase:
        owns += f" && {dense.spin_of('head', 'm_dense')} < 0"
    known = {"kind": ROW_FRICTION, "parent": "head", "triple_phase": triple_phase}
    pair = f"{pad}if ({owns}) {{\n{visit('head + 1', pad + '    ', **known)}\n{pad}}}"
    levels = Levels(
        count,
        offsets.format,
        f"const int head = sparse_contact_level_units.data[world * {M} + place];\n"
        f"const int unit_coordinate = {op.index('head', '0')};",
        stride="STRIDE",
    )
    # A head is a contact's normal or a joint limit (the builder's): each steps by its kind's law, compiled apart.
    contact, limit = visit("head", pad + "    ", kind=ROW_CONTACT), visit("head", pad + "    ", kind=ROW_JOINT_LIMIT)
    heads = (
        f"{pad}if (({dense.kind('head')}) == {ROW_CONTACT}) {{\n{contact}\n{indent(pair, '    ')}\n"
        f"{pad}}} else {{\n{limit}\n{pad}}}"
    )
    phase = f"        {{\n{levels.wrap(heads, ' ' * 12)}\n        }}"
    # The marks the stage holds: the head's rows (every leveled head is a live row of the world, which the head holds).
    marks = f"""    for (int w = lane; w < {(dense.staged_rows + 31) // 32}; w += STRIDE) s_level_mark[w] = 0;
    SYNC();
    for (int p = lane; p < {offsets.format(count)}; p += STRIDE) {{
        const int head = sparse_contact_level_units.data[world * {M} + p];
        atomicOr((unsigned*)&s_level_mark[head >> 5], 1u << (head & 31));
    }}
    SYNC();"""
    member = f"({dense.kind('i')}) == {ROW_FRICTION}" + (
        f" || ({dense.kind('i')}) == {ROW_TORSION}" if laws.torsion else ""
    )
    skip = (
        f"const int unit_head = {member} ? {dense.parent('i', enclosed=True)} : i;\n"
        "if ((s_level_mark[unit_head >> 5] >> (unit_head & 31)) & 1) continue;"
    )
    return marks, phase, skip


_PAIR_BLOCK = """if ((kind & 16) != 0) {
    // The group's block over its six relative coordinates y = t_a - P t_b, whitened by W = L L^T
    // (yh = L^-1 y), from its staged records (region_sweep.stage_blocks), by the region law's block laws
    // (region_laws.block_*): each member's friction unit by the region law's cone law at the load its region holds,
    // then every member candidate's normal as one compliant block (the region law's closing springs, Newton on the
    // six). The held wrenches are kept whitened (L^T J). A block takes its world's lanes (teamed: region_phase), so
    // its member batches and candidate passes split over them (team).
    const int team = STRIDE, mine = lane;
    const int bid = world * 2 + (kind >> 8);
    const wp::vec2i pair = blk_pair.data[bid];
    const bool flip = pair[0] != f;
    wp::spatial_vector& ua = flip ? u_b : u;
    wp::spatial_vector& ub = flip ? u : u_b;
    int& ver_a = flip ? version_b : version;
    int& ver_b = flip ? version : version_b;
    const int fa = pair[0], fb = pair[1];
    const int first_pa = flip ? first_b : first, first_pb = flip ? first : first_b;
    const wp::spatial_vector* ba = body_basis.data + (size_t)fa * 6;
    const wp::spatial_vector* bb = body_basis.data + (size_t)(fb >= 0 ? fb : fa) * 6;
    const wp::vec3 dd = blk_d.data[bid];
    const wp::spatial_matrix Lm = blk_L.data[bid];
    float ta[6], tb[6];
    for (int k = 0; k < 6; ++k) { ta[k] = wp::dot(ba[k], ua); tb[k] = fb >= 0 ? wp::dot(bb[k], ub) : 0.0f; }
    const wp::vec3 wbd = wp::cross(wp::vec3(tb[3], tb[4], tb[5]), dd);
    wp::spatial_vector y;
    for (int k = 0; k < 3; ++k) {
        y[k] = ta[k] - tb[k] - wbd[k];
        y[k + 3] = ta[k + 3] - tb[k + 3];
    }
    wp::spatial_vector yh = RegionLaw::block_enter(Lm, y);
    const wp::spatial_vector Jn0 = blk_wrench.data[2 * bid], Jf0 = blk_wrench.data[2 * bid + 1];
    wp::spatial_vector Jf = Jf0;
    const wp::vec4i blob = blk_blob.data[bid];
    float* const stage = blk_stage.data;
    const int* const istage = blk_istage.data;
    float* const lambda = world_impulses.data;
    bool slid = false;
    if (team > 1) WARP_SYNC();
    // The units, member by member, at the loads the members hold (each the sum of its candidates' impulses): the
    // team's lanes read a batch of members at once (their loads, whether they hold anything), then take the batch's
    // live members in order. The lane that read a member's load keeps its new friction and stores it after the batch,
    // so no store stands between two units (no unit reads another's friction).
    for (int base = 0; base < blob[1]; base += team) {
        bool live = false;
        float own_load = 0.0f;
        if (base + mine < blob[1]) {
            const auto member = RegionLaw::block_load(blk_stage, blob[0] + 44 * (base + mine), blob[2]);
            own_load = member.v0;
            live = member.v1;
        }
        unsigned bits = live ? 1u : 0u;
        if (team > 1) {
            bits = BALLOT(live);
            WARP_SYNC();
        }
        wp::vec3 own_phi;
        bool own_set = false;
        while (bits != 0u) {
            const int j = FIRST(bits);
            bits &= bits - 1u;
            const float member_load = team > 1 ? READ(own_load, j) : own_load;
            const auto unit = RegionLaw::block_unit(blk_stage, blob[0] + 44 * (base + j), yh, member_load, slid);
            yh = wp::add(yh, unit.v1);
            Jf = wp::add(Jf, unit.v1);
            if (mine == j) {
                own_phi = unit.v0;
                own_set = true;
            }
            slid = unit.v2;
        }
        if (own_set) {
            const int at = blob[0] + 44 * (base + mine);
            stage[at + 39] = own_phi[0];
            stage[at + 40] = own_phi[1];
            stage[at + 41] = own_phi[2];
            const int row = istage[at];
            if (row >= 0) {
                lambda[row] = own_phi[0];
                lambda[row + 1] = own_phi[1];
                if ((int)stage[at + 23] == 3) lambda[row + 2] = own_phi[2];
            }
        }
    }
    // The normals as one compliant block: their own impulses leave, Newton on the six coordinates; the candidates
    // over the team's lanes, the sums reduced over them.
    const wp::spatial_vector ys = wp::sub(yh, Jn0);
    int last_count = -1, last_sum = 0, last_square = 0;
    for (int step = 0; step < 16; ++step) {
        wp::spatial_matrix S;
        wp::spatial_vector q;
        int count = 0, index_sum = 0, index_square = 0;
        for (int c = mine; c < blob[3]; c += team) {
            const int at = blob[2] + 12 * c;
            const auto cand = RegionLaw::block_candidate(blk_stage, at, yh, S, q);
            S = cand.v0;
            q = cand.v1;
            if (cand.v3) {
                count += 1;
                index_sum += c;
                index_square += c * c;
            }
            stage[at + 8] = cand.v2;
            lambda[istage[at]] = cand.v2;
        }
        if (team > 1) {
            for (int a = 0; a < 6; ++a)
                for (int b = 0; b <= a; ++b) SUM(S.data[a][b]);
            for (int k = 0; k < 6; ++k) SUM(q[k]);
            SUM(count, index_sum, index_square);
        }
        if (count == last_count && index_sum == last_sum && index_square == last_square) break;
        last_count = count;
        last_sum = index_sum;
        last_square = index_square;
        yh = RegionLaw::block_newton(S, q, ys);
    }
    if (team > 1) WARP_SYNC();
    // The held wrenches (whitened) and their change, back to the bodies: J = L^-T (L^T J).
    const auto leave = RegionLaw::block_leave(Lm, yh, ys, Jf, Jn0, Jf0);
    const wp::spatial_vector held = leave.v1, own = leave.v2, dJ = leave.v3;
    const wp::vec3 dF(dJ[0], dJ[1], dJ[2]);
    const wp::vec3 dT = wp::add(wp::vec3(dJ[3], dJ[4], dJ[5]), wp::cross(dd, dF));
    const float dJb[6] = {dJ[0], dJ[1], dJ[2], dT[0], dT[1], dT[2]};
    auto push_body = [&](wp::spatial_vector& uu, const wp::spatial_vector* bs, const float* J, const float sgn,
                         const int fst, const int ff, int& ver) {
        bool moved = false;
        for (int k = 0; k < 6; ++k) {
            const float now = uu[k] + sgn * (bs[0][k] * J[0] + bs[1][k] * J[1] + bs[2][k] * J[2]
                                             + bs[3][k] * J[3] + bs[4][k] * J[4] + bs[5][k] * J[5]);
            moved = moved || now != uu[k];
            uu[k] = now;
            s_v[fst + k] = now;
        }
        if (moved) {
            ver += 1;
            body_version.data[ff] = ver;
            body_seen.data[ff] = uu;
        }
    };
    push_body(ua, ba, dJ.c, 1.0f, first_pa, fa, ver_a);
    if (fb >= 0) push_body(ub, bb, dJb, -1.0f, first_pb, fb, ver_b);
    v = wp::vec3(wp::dot(basis[0], u), wp::dot(basis[1], u), wp::dot(basis[2], u));
    w = wp::vec3(wp::dot(basis[3], u), wp::dot(basis[4], u), wp::dot(basis[5], u));
    blk_wrench.data[2 * bid] = leave.v0;
    blk_wrench.data[2 * bid + 1] = Jf;
    bool changed = slid;
    for (int k = 0; k < 6; ++k) {
        changed = changed || own[k] != 0.0f;
        if (!(fabsf(own[k]) <= QUIET_ * (fabsf(yh[k]) + fabsf(held[k])))) iteration_changed = 1;
    }
    // Settled only by a visit that changed nothing (the units read the loads the normals leave).
    reg_state.data[g] = changed ? (-1) : version;
    reg_stamp.data[g] = version_b;
    if (team > 1) WARP_SYNC();
    continue;
}
"""
"""The native visit of a group's block (:func:`.region_sweep.form_blocks`), at the region that stands for it; ``QUIET_``
stands for the exit law's floor."""


REGION_OPERANDS = (
    ("world_regions", wp.array[wp.vec2i]),
    ("world_levels", wp.array[int]),
    ("world_order_end", wp.array[int]),
    ("level_end", wp.array[int]),
    ("level_region", wp.array[int]),
    ("reg_order", wp.array[int]),
    ("reg_walk", wp.array[int]),
    ("reg_level", wp.array[int]),
    ("reg_rank", wp.array[int]),
    ("reg_after", wp.array[wp.vec2i]),
    ("body_level", wp.array[int]),
    ("body_first", wp.array[int]),
    ("body_basis", wp.array2d[wp.spatial_vector]),
    ("reg_frame", wp.array[wp.mat33]),
    ("reg_arm", wp.array[wp.vec3]),
    ("reg_arm_b", wp.array[wp.vec3]),
    ("reg_body", wp.array[wp.vec2i]),
    ("reg_cand", wp.array[wp.vec2i]),
    ("reg_law", wp.array[wp.vec2]),
    ("reg_bf", wp.array[wp.vec3]),
    ("reg_kind", wp.array[int]),
    ("reg_row", wp.array[int]),
    ("reg_wnn", wp.array[wp.mat33]),
    ("reg_wnf", wp.array[wp.mat33]),
    ("reg_wff", wp.array[wp.mat33]),
    ("reg_ball_values", wp.array[wp.vec3]),
    ("reg_ball_vectors", wp.array[wp.mat33]),
    ("reg_ginv", wp.array[wp.mat33]),
    ("reg_k", wp.array[wp.mat33]),
    ("reg_lk", wp.array[wp.mat33]),
    ("reg_lw", wp.array[wp.mat33]),
    ("cand", wp.array[wp.vec4]),
    ("cand_floor", wp.array[float]),
    ("cand_row", wp.array[int]),
    ("reg_wrench", wp.array[wp.vec3]),
    ("reg_phi", wp.array[wp.vec3]),
    ("reg_set", wp.array[wp.vec3i]),
    ("reg_spring", wp.array[wp.vec3]),
    ("reg_state", wp.array[int]),
    ("reg_stamp", wp.array[int]),
    ("reg_bound", wp.array[wp.vec2]),
    ("body_seen", wp.array[wp.spatial_vector]),
    ("body_version", wp.array[int]),
    ("world_free_start", wp.array[int]),
    ("world_free", wp.array[int]),
    ("body_end_twist", wp.array[wp.spatial_vector]),
    ("blk_pair", wp.array[wp.vec2i]),
    ("blk_blob", wp.array[wp.vec4i]),
    ("blk_L", wp.array[wp.spatial_matrix]),
    ("blk_d", wp.array[wp.vec3]),
    ("blk_wrench", wp.array[wp.spatial_vector]),
    ("blk_stage", wp.array[float]),
    ("blk_istage", wp.array[int]),
    ("region_cfm", float),
)
"""The operands of a sweep that visits regions (:mod:`.region_sweep`'s stream): each world's open regions, each free
body's first world coordinate and basis, the regions' records, step terms (:func:`.region_sweep.region_terms`),
candidates and state, each open
body's walk coordinates as its last visit or sweep left them with its version (``body_seen``, ``body_version``: the
pass rule's; ``reg_stamp`` a region's second body's version at its last settled visit), each world's free bodies and
their twists at a sweep's end (``world_free_start``, ``world_free``, ``body_end_twist``: the sweeps' end test's,
:func:`region_verify`), the friction units' regularization."""


def region_phase(pad: str = "        ", *, block: bool = False, group: bool = False) -> str:
    """Native statements of a sweep's region phase: the world's open regions (those of a free body that a dense row,
    or a region between two movable bodies touches; :func:`.region_sweep.fill_regions`), body by body in
    their order, each by the one visit (:func:`.region_laws.region_visit`), on the world's velocity change in walk
    coordinates (``s_v``).

    The order is taken by its exact levels over the regions that can act (:func:`.region_sweep.level_acting`): a
    level's regions share no body, so the world's lanes visit them at once, and a barrier ends a level; that is the
    sequential sweep itself. The left-out regions follow them in the walk and take no level: a sweep passes them
    anyway, and one the verify opens takes its place in the levels made anew (:func:`region_verify`). A
    world whose levels hold fewer regions than the colored rule's parallelism (a chain: a tangled pair's regions
    share both bodies, one a level) is walked by its lanes together in the order the levels were made from
    (``reg_walk``: a body's regions together within a depth), a first body's twist kept from one of its regions to
    the next, as a sequential walk does; the result is the same either way. The walk reads its order's states a
    lanes' width at a time and takes the regions that are not left out (a left-out region's turn writes nothing). A pair block (:func:`.region_sweep.form_blocks`)
    takes every lane of its world either way: its member batches and candidate passes split over them, so its sums
    reduce in one order whatever its world's walk. A region's response and the other terms
    that stand for the step are the listing's (:func:`.region_sweep.region_terms`), and its wrench's change returns
    through its bodies' bases.

    The pass rule is exact, by a version a body: the version is raised whenever a bit of the body's walk coordinates
    differs from those its last visit, or its last sweep, left (``body_seen``). A region that holds a wrench and
    whose last visit settled it is stamped with its bodies' versions (``reg_state`` its first body's, ``reg_stamp``
    its second's) and passed while both stand; one that holds nothing is opened only if a candidate closes at its
    bodies' twists. A visit votes for another sweep by the exit law: a coordinate of the region whose own change is
    past the floor (:data:`.solve_policy.QUIET`) of its terms. A left-out region
    (:data:`.region_sweep.REGION_LEFT`) is passed: the sweeps' end tests it (:func:`region_verify`).

    Over a world's ``group`` of blocks (:func:`worlds`' ranks), the world's block (rank 0) runs the phase as a block's,
    and a level that holds more regions than the block has threads is the group's: the block posts the level's places
    in the world's job words (:data:`WIDE_JOB`), every block of the group takes its regions (spread over the blocks,
    ``tid * group + rank``), and the block joins them, with their exit votes, before its next level. The other blocks
    take what it posts until the world's end. A level's regions share no body, so the order is the same sweep."""
    # A body's version from its words as the region's head loaded them (``{u}_in``, ``{seen}``, ``{version}_in``): raised
    # where its walk coordinates differ from those it was last seen at.
    # Its store waits for the twists (the visit head below), so no store stands between the head's loads and the bases'.
    gather = """{u} = {u}_in;
{version} = {version}_in;
bool same = true;
for (int k = 0; k < 6; ++k) same = same && {u}[k] == {seen}[k];
{raised} = !same;
if ({raised}) {version} += 1;"""
    push = """bool moved = false;
for (int k = 0; k < 6; ++k) {{
    const float now = {u}[k] {op} sign * ({basis}[0][k] * {wrench}.v0[0] + {basis}[1][k] * {wrench}.v0[1] + {basis}[2][k] * {wrench}.v0[2]
                                     + {basis}[3][k] * {wrench}.v1[0] + {basis}[4][k] * {wrench}.v1[1] + {basis}[5][k] * {wrench}.v1[2]);
    moved = moved || now != {u}[k];
    {u}[k] = now;
    s_v[{first} + k] = now;
}}
if (moved) {{
    {version} += 1;
    body_version.data[{f}] = {version};
    body_seen.data[{f}] = {u};
}}"""
    first_body = indent(gather.format(u="u", seen="seen", version="version", raised="raised"), "            ")
    second_body = indent(gather.format(u="u_b", seen="seen_b", version="version_b", raised="raised_b"), "            ")
    push_a = indent(
        push.format(u="u", op="+", basis="basis", wrench="push", first="first", version="version", f="f"),
        "            ",
    )
    push_b = indent(
        push.format(u="u_b", op="-", basis="basis_b", wrench="push_b", first="first_b", version="version_b", f="f_b"),
        "            ",
    )
    parallelism = repr(float(sweep_units.COLORED_RULE.parallelism))
    pair_block = "" if block else indent(_PAIR_BLOCK.replace("QUIET_", QUIET_LITERAL), "        ")
    # Over a block (the colored mapping's, a world whose levels hold the colored rule's parallelism and that forms no
    # pair block: its claim's rule), a level's regions take every thread at the block's stride and a block barrier
    # ends a level; no world there is a chain and no turn is teamed.
    if group and not block:
        raise ValueError("a world's group of blocks serves the block's levels")
    sync, me, stride = ("__syncthreads();", "tid", "(int)blockDim.x") if block else ("SYNC();", "lane", "STRIDE")
    if group:
        me, stride = "place_at", "place_step"
    alone = "false" if block else f"(float)(kept_end - region_span[0]) < {parallelism}f * (float)levels"
    holds = "false" if block else f"blk_pair.data[world * {int(PAIR_BLOCKS)}][0] >= 0"
    passed = "place >= level_last" if block else "(place >= level_last || ((blocks >> lane) & 1u) != 0u)"
    visit = f"""        const int g = alone ? reg_walk.data[place] : level_region.data[place];
        // The region's words at once (its state, bodies, second body's stamp, kind): one round of memory latency.
        const int state = reg_state.data[g];
        const wp::vec2i sides = reg_body.data[g];
        const int stamp = reg_stamp.data[g];
        const int kind = reg_kind.data[g];
        // A left-out region waits for the sweeps' end (region_verify).
        if (state == {int(REGION_LEFT)}) continue;
        const int f = sides[0], f_b = sides[1];
        // Both bodies' words at once (a lone body reads its own twice): their first coordinates, the coordinates
        // their versions were taken at and the versions, then their walk coordinates. A body's version is raised
        // only by its own gather, so the second body's words do not wait for the first's.
        const int other = f_b >= 0 ? f_b : f;
        const int first_in = body_first.data[f], first_b_in = body_first.data[other];
        const wp::spatial_vector seen = body_seen.data[f], seen_b = body_seen.data[other];
        const int version_in = body_version.data[f], version_b_in = body_version.data[other];
        wp::spatial_vector u_in, u_b_in;
        for (int k = 0; k < 6; ++k) {{
            u_in[k] = s_v[first_in + k];
            u_b_in[k] = s_v[first_b_in + k];
        }}
        bool raised = false, raised_b = false;
        if (!alone || f != body) {{
            // The first body: its walk coordinates, its version and its twist.
            body = f;
            first = first_in;
            basis = body_basis.data + (size_t)f * 6;
            {{
{first_body}
            }}
            v = wp::vec3(wp::dot(basis[0], u), wp::dot(basis[1], u), wp::dot(basis[2], u));
            w = wp::vec3(wp::dot(basis[3], u), wp::dot(basis[4], u), wp::dot(basis[5], u));
        }}
        // A region between two movable bodies: its second body's walk coordinates, version and twist.
        int first_b = -1, version_b = 0;
        wp::spatial_vector u_b;
        const wp::spatial_vector* basis_b = basis;
        wp::vec3 v_b, w_b;
        if (f_b >= 0) {{
            first_b = first_b_in;
            basis_b = body_basis.data + (size_t)f_b * 6;
            {{
{second_body}
            }}
            v_b = wp::vec3(wp::dot(basis_b[0], u_b), wp::dot(basis_b[1], u_b), wp::dot(basis_b[2], u_b));
            w_b = wp::vec3(wp::dot(basis_b[3], u_b), wp::dot(basis_b[4], u_b), wp::dot(basis_b[5], u_b));
        }}
        // The gathers' stores, after the twists.
        if (raised) {{
            body_version.data[f] = version;
            body_seen.data[f] = u;
        }}
        if (raised_b) {{
            body_version.data[f_b] = version_b;
            body_seen.data[f_b] = u_b;
        }}
        if (state == version && (f_b < 0 || stamp == version_b)) continue;
{pair_block}
        const float sign = (kind & {int(KIND_SECOND)}) != 0 ? -1.0f : 1.0f;
        const wp::mat33 frame = reg_frame.data[g];
        const wp::vec3 n(frame.data[0][0], frame.data[0][1], frame.data[0][2]);
        const wp::vec3 t0(frame.data[1][0], frame.data[1][1], frame.data[1][2]);
        const wp::vec3 t1(frame.data[2][0], frame.data[2][1], frame.data[2][2]);
        const wp::vec3 arm = reg_arm.data[g], arm_b = reg_arm_b.data[g];
        const auto twist = RegionLaw::frame_twist(v, w, arm, n, t0, t1);
        wp::vec3 sn = wp::mul(twist.v0, sign), sf = wp::mul(twist.v1, sign);
        if (f_b >= 0) {{
            const auto twist_b = RegionLaw::frame_twist(v_b, w_b, arm_b, n, t0, t1);
            sn = wp::sub(sn, wp::mul(twist_b.v0, sign));
            sf = wp::sub(sf, wp::mul(twist_b.v1, sign));
        }}
        const wp::vec2i span = reg_cand.data[g];
        if (state == {int(REGION_IDLE)}) {{
            // A region that holds nothing is opened only if a candidate closes; its closing bound passes it without
            // the candidates' reads where none can (region_laws.stays_open, exact: the test below would leave it).
            if (RegionLaw::stays_open(sn, reg_bound.data[g])) continue;
            bool closes = false;
            for (int i = span[0]; i < span[1]; ++i) {{
                const wp::vec4 record = cand.data[i];
                closes = closes || RegionLaw::candidate_closes(record, sn, cand_floor.data[i]);
            }}
            if (!closes) continue;
        }}
        const wp::vec2 law = reg_law.data[g];
        const wp::vec3i set_in = reg_set.data[g];
        // The region's terms are read where they are used (RegionLaw::reread), not held across the visit.
        const auto out = RegionLaw::region_visit(
            sn, sf, g, reg_wnn, reg_wnf, reg_wff, region_cfm, reg_bf.data[g], law[0], law[1],
            kind & {int(FRICTION_ROWS)}, reg_ball_values, reg_ball_vectors, reg_ginv, reg_k, reg_lk, reg_lw, cand,
            cand_floor, span[0], span[1], reg_wrench.data[g], reg_phi.data[g], set_in[0], set_in[1], set_in[2],
            reg_spring.data[g]);
        const int again = RegionLaw::reread(g);
        // The change of the region's wrench through its bodies' bases.
        const wp::vec3 d_wrench = wp::sub(out.v0, reg_wrench.data[again]), d_phi = wp::sub(out.v1, reg_phi.data[again]);
        const wp::mat33 frame_out = reg_frame.data[again];
        const wp::vec3 n_out(frame_out.data[0][0], frame_out.data[0][1], frame_out.data[0][2]);
        const wp::vec3 t0_out(frame_out.data[1][0], frame_out.data[1][1], frame_out.data[1][2]);
        const wp::vec3 t1_out(frame_out.data[2][0], frame_out.data[2][1], frame_out.data[2][2]);
        {{
            const auto push = RegionLaw::frame_wrench(d_wrench, d_phi, reg_arm.data[again], n_out, t0_out, t1_out);
{push_a}
            if (moved) {{
                v = wp::vec3(wp::dot(basis[0], u), wp::dot(basis[1], u), wp::dot(basis[2], u));
                w = wp::vec3(wp::dot(basis[3], u), wp::dot(basis[4], u), wp::dot(basis[5], u));
            }}
        }}
        if (f_b >= 0) {{
            const auto push_b = RegionLaw::frame_wrench(d_wrench, d_phi, reg_arm_b.data[again], n_out, t0_out, t1_out);
{push_b}
        }}
{_FLOW_MARK}
        // The exit law's vote: a coordinate of the region whose own change is past the floor of its terms (its
        // velocity, its right-hand side, what the region's wrench holds of it).
        {{
            const wp::vec3 rhs_f = reg_bf.data[again];
            const wp::mat33 response_nn = reg_wnn.data[again], response_nf = reg_wnf.data[again];
            const wp::mat33 response_ff = reg_wff.data[again];
            const wp::mat33 coupling = wp::transpose(response_nf);
            const wp::vec3 own_n = wp::add(wp::mul(response_nn, d_wrench), wp::mul(response_nf, d_phi));
            const wp::vec3 own_f = wp::add(wp::mul(coupling, d_wrench), wp::mul(response_ff, d_phi));
            const wp::vec3 held_n = wp::add(wp::mul(response_nn, out.v0), wp::mul(response_nf, out.v1));
            const wp::vec3 held_f = wp::add(wp::mul(coupling, out.v0), wp::mul(response_ff, out.v1));
            for (int k = 0; k < 3; ++k) {{
                if (!(fabsf(own_n[k]) <= {QUIET_LITERAL} * (fabsf(sn[k]) + fabsf(held_n[k])))) iteration_changed = 1;
                if (!(fabsf(own_f[k]) <= {QUIET_LITERAL} * (fabsf(sf[k]) + fabsf(rhs_f[k]) + fabsf(held_f[k]))))
                    iteration_changed = 1;
            }}
        }}
        // The candidates' impulses, the friction's, the region's state.
        float* const lambda = world_impulses.data;
        const wp::vec3i set_old = reg_set.data[again];
        for (int k = 0; k < 3; ++k) if (set_old[k] >= 0) lambda[cand_row.data[set_old[k]]] = 0.0f;
        const int set_new[3] = {{out.v2, out.v3, out.v4}};
        for (int k = 0; k < 3; ++k) if (set_new[k] >= 0) lambda[cand_row.data[set_new[k]]] = out.v5[k];
        if ((kind & {int(KIND_SPRINGS)}) != 0) {{
            for (int i = span[0]; i < span[1]; ++i) {{
                const wp::vec4 record = cand.data[i];
                if (record[3] != 0.0f) {{
                    const wp::vec3 a = RegionLaw::candidate_row(record);
                    const float closing = wp::min(wp::dot(a, out.v7) + record[2], 0.0f);
                    lambda[cand_row.data[i]] = -closing / record[3];
                }}
            }}
        }}
        const int friction_row = reg_row.data[g];
        if (friction_row >= 0) {{
            lambda[friction_row] = out.v1[0];
            lambda[friction_row + 1] = out.v1[1];
            if ((kind & {int(FRICTION_ROWS)}) == 3) lambda[friction_row + 2] = out.v1[2];
        }}
        reg_wrench.data[g] = out.v0;
        reg_phi.data[g] = out.v1;
        reg_set.data[g] = wp::vec3i(out.v2, out.v3, out.v4);
        reg_spring.data[g] = out.v6;
        reg_state.data[g] = out.v8 == {int(REGION_SLIDE)} ? {int(REGION_LOOSE)} : out.v0[0] > 0.0f ? version : {int(REGION_IDLE)};
        reg_stamp.data[g] = version_b;
"""
    if block and not group:
        return indent(_region_flow(visit.replace(_FLOW_MARK, _FLOW_RAN)), pad)
    visit = visit.replace(_FLOW_MARK + "\n", "")
    text = f"""{sync}
{{
    const wp::vec2i region_span = world_regions.data[world];
    const int levels = world_levels.data[world];
    // The colored rule's measure (sweep_units.COLORED_RULE): a world whose levels hold fewer regions than its
    // parallelism, a chain, is walked by its first lane in the order the levels were made from, a body's twist kept
    // from one region to the next of the same body; the others' levels take the lanes. Both are the same sweep.
    const int kept_end = levels > 0 ? level_end.data[region_span[0] + levels - 1] : region_span[0];
    const bool alone = {alone};
    // Whether the world holds a pair block: its first is listed first (region_sweep.form_blocks).
    const bool holds_block = {holds};
    int level_first = region_span[0];
    int body = -1, first = 0, version = 0;
    wp::spatial_vector u;
    wp::vec3 v, w;
    const wp::spatial_vector* basis = body_basis.data;
    // A chain world walks its order itself (reg_walk: a body's regions together, so its twist stays kept), as one
    // span; the others take the levels.
    const int spans = alone ? 1 : levels;
{_REGION_GROUP_STAGE if group else _LEVEL_STAGE}
    for (int chunk = level_first; chunk < level_last; chunk += {stride}) {{
    // A chain world's lanes all walk its order, each doing the same work (a block's candidate passes split over them),
    // a chunk of the lanes' width at a time: each lane reads one place's state, and the chunk's turns are its regions
    // that are not left out, in order (a left-out region stays so until the sweeps' end and its turn writes nothing).
    // A pair block takes every lane (teamed): the lanes visit a chunk's blocks together, one after another, then each
    // lane its own other region; a level's regions share no body, so its order inside the level is the same sweep.
    const unsigned blocks = alone ? BALLOT(chunk + lane < level_last
        && reg_state.data[reg_walk.data[chunk + lane]] != {int(REGION_LEFT)}) : !holds_block ? 0u
        : BALLOT(chunk + lane < level_last && (reg_kind.data[level_region.data[chunk + lane]] & {int(KIND_BLOCK)}) != 0);
    unsigned left = blocks;
    const int turns = alone ? COUNT(blocks) : COUNT(blocks) + 1;
    // One copy of a region's visit: the turns are a schedule, not to be unrolled (each copy is the whole visit).
    #pragma unroll 1
    for (int turn = 0; turn < turns; ++turn) {{
        const bool teamed = alone || left != 0u;
        const int place = chunk + (left != 0u ? FIRST(left) : {me});
        left &= left - 1u;
        if (!teamed && {passed}) continue;
        if (teamed) WARP_SYNC();
{visit}    }}
    }}
{GROUP_JOIN if group else ""}    level_first = level_last;
    if (!alone) {sync}
    }}
}}
{sync}"""
    return indent(text, pad)


_FLOW_MARK = "        // FLOW_RAN"
"""Where a region's visit has left its bodies' coordinates and versions (after its pushes): the flow marks its turn
there (:data:`_FLOW_RAN`), so the next region on each body waits for no store of the region's own state."""

_FLOW_RAN = """        // Its bodies hold their coordinates and versions: its turn on them, so the next region on each may run while
        // this one stores its own state.
        __threadfence_block();
        {
            volatile int* const done = s_ran;
            done[slot_a] = region_turn + flow_level;
            if (flow_sides[1] >= 0) done[slot_b] = region_turn + flow_level;
            marked = true;
        }"""


def _region_flow(visit: str) -> str:
    """The region phase over a world's block (:func:`region_phase`'s block form) as a flow: the block takes the acting
    regions by level in rounds of its threads, a thread a place, and a region runs once the regions before it on its
    bodies ran (the levels its bodies held before it in the walk, ``reg_after``: the level builder's), each body's
    latest region's turn in ``s_ran`` at its first coordinate's slot (:data:`RAN_SLOT`). A level's regions share no
    body and a body's regions run in the walk's order, so every region reads the coordinates the sequential sweep of
    the order gives it: the levels' sweep, without a barrier a level nor a level's slowest visit. No region waits
    forever: those it waits for sit at lower levels, at earlier places, and a warp takes its rounds in order and
    polls until its lanes ran. The turns rise by the levels each sweep (``region_turn``), so ``s_ran`` needs no
    clearing between sweeps, nor after the sweeps' end remakes the levels."""
    return f"""__syncthreads();
{{
    const wp::vec2i region_span = world_regions.data[world];
    const int levels = world_levels.data[world];
    const int kept_end = levels > 0 ? level_end.data[region_span[0] + levels - 1] : region_span[0];
    const bool alone = false;
    for (int base = region_span[0]; base < kept_end; base += (int)blockDim.x) {{
        const int place = base + tid;
        bool pending = place < kept_end;
        // The place's region, its level and the levels its bodies held before it, and its bodies' slots.
        const int at = pending ? level_region.data[place] : level_region.data[region_span[0]];
        const wp::vec2i flow_sides = reg_body.data[at];
        const wp::vec2i after = reg_after.data[at];
        const int flow_level = reg_level.data[at];
        const int slot_a = body_first.data[flow_sides[0]] / {RAN_SLOT};
        const int slot_b = flow_sides[1] >= 0 ? body_first.data[flow_sides[1]] / {RAN_SLOT} : slot_a;
        while (__any_sync(0xffffffffu, pending)) {{
            if (pending) {{
                volatile const int* const ran = s_ran;
                const bool ready = (after[0] < 0 || ran[slot_a] == region_turn + after[0])
                    && (after[1] < 0 || ran[slot_b] == region_turn + after[1]);
                if (ready) {{
                    __threadfence_block();
                    int body = -1, first = 0, version = 0;
                    wp::spatial_vector u;
                    wp::vec3 v, w;
                    const wp::spatial_vector* basis = body_basis.data;
                    // A visit marks its turn once its pushes are done (_FLOW_RAN); a region passed before them here.
                    bool marked = false;
                    do {{
{indent(visit, "                ")}
                    }} while (false);
                    if (!marked) {{
                        __threadfence_block();
                        volatile int* const done = s_ran;
                        done[slot_a] = region_turn + flow_level;
                        if (flow_sides[1] >= 0) done[slot_b] = region_turn + flow_level;
                    }}
                    pending = false;
                }}
            }}
        }}
    }}
    region_turn += levels;
}}
__syncthreads();"""


# A world's job words where it takes a group of blocks (region_phase): the generation of its last post, the level's
# first and last places, the world's end, the blocks that joined, their exit votes.
WIDE_JOB = 8

_VERIFY_TURNS = 4
"""The places a lane of the sweeps' end test (:func:`region_verify`) loads at once: their words (the region's record
and its bodies' twists) in flight together, not a test's latency each. A hardware choice that changes no result:
each test reads only its own region and the bodies' twists, and writes only its own state."""

_LEVEL_STAGE = """    for (int level = 0; level < spans; ++level) {
    const int level_last = alone ? kept_end : level_end.data[region_span[0] + level];"""


def group_stage(index: str, first: str, last: str, take: str, slots: str, me: str, declare: str = "") -> str:
    """The head of a stage over a world's group of blocks (:func:`worlds`' ranks): the world's block (rank 0) takes its
    next stage (``take``: sets ``first`` and ``last``, or leaves the loop at the world's last) and posts one wider than
    its ``slots`` places (a block's threads for regions, its warps for groups of rows) in the world's job words
    (:data:`WIDE_JOB`); the other blocks take what it posts, the world's end last (``wide_gen`` counts the posts on
    every block of the group). A posted stage's places are spread over the group's blocks (``me * group + rank``),
    a narrow one's over the block; :data:`GROUP_JOIN` ends a posted one. ``declare`` opens each stage."""
    return f"""    for (int {index} = 0; ; ++{index}) {{
{declare}
    bool wide = true;
    if (rank == 0) {{
{take}
        wide = group > 1 && {last} - {first} > {slots};
        if (wide) {{
            ++wide_gen;
            if (tid == 0) {{
                volatile int* const job = wide_job;
                job[1] = {first};
                job[2] = {last};
                __threadfence();
                atomicExch(wide_job, wide_gen);
            }}
        }}
    }} else {{
        ++wide_gen;
        if (tid == 0) {{
            const volatile int* const job = wide_job;
            while (job[0] < wide_gen) {{ }}
            __threadfence();
        }}
        __syncthreads();
        const volatile int* const job = wide_job;
        if (job[3] != 0) break;
        {first} = job[1];
        {last} = job[2];
    }}
    const int place_at = wide ? {me} * group + rank : {me};
    const int place_step = wide ? {slots} * group : {slots};"""


# The join of a posted level: the world's block waits for the group's blocks and takes their exit votes; each of them
# votes, then counts itself in.
GROUP_JOIN = """    if (wide) {
        if (rank == 0) {
            __syncthreads();
            if (tid == 0) {
                volatile int* const job = wide_job;
                while (job[4] < wide_gen * (group - 1)) { }
                __threadfence();
                if (job[5] != 0) {
                    job[5] = 0;
                    iteration_changed = 1;
                }
            }
        } else {
            const bool voted = __syncthreads_or(iteration_changed != 0);
            iteration_changed = 0;
            if (tid == 0) {
                if (voted) atomicOr(wide_job + 5, 1);
                __threadfence();
                atomicAdd(wide_job + 4, 1);
            }
        }
    }
"""


# The region phase's stage over a world's group of blocks: the world's levels in turn, a level a block's threads.
_REGION_GROUP_STAGE = group_stage(
    "level",
    "level_first",
    "level_last",
    "        if (level >= spans) break;\n        level_last = level_end.data[region_span[0] + level];",
    "(int)blockDim.x",
    "tid",
    declare="    int level_last = level_first;",
)


def region_verify(
    iteration: str = ITERATION, budget: str = "sweep_budget", pad: str = "        ", *, block: bool = False
) -> str:
    """Native statements of a sweep's end that test the world's left-out regions (:data:`.region_sweep.REGION_LEFT`:
    none of their candidates closed at the predictor): the world's lanes take its regions, each left-out one at its
    bodies' twists as the sweep leaves them (their walk coordinates ``s_v`` through their bases) by the phase's own
    test of a region that holds nothing (its closing bound, then the closing test where the bound cannot rule it
    out). One that
    closes joins the next sweep as a region that holds nothing (:data:`.region_sweep.REGION_IDLE`) at its place in
    the order: where one joins, the world's walk and its levels are made anew over the regions that can act (the
    statements of :func:`.region_sweep.level_acting`), and the sweeps may not stop; a sweep more where ``budget`` ends
    at ``iteration``. So every region still left out at the end holds
    nothing at a motion where none of its candidates closes."""
    twist = """const int first{s} = body_first.data[{f}];
const wp::spatial_vector* const basis{s} = body_basis.data + (size_t){f} * 6;
wp::spatial_vector u{s};
for (int k = 0; k < 6; ++k) u{s}[k] = s_v[first{s} + k];
const wp::vec3 v{s}(wp::dot(basis{s}[0], u{s}), wp::dot(basis{s}[1], u{s}), wp::dot(basis{s}[2], u{s}));
const wp::vec3 w{s}(wp::dot(basis{s}[3], u{s}), wp::dot(basis{s}[4], u{s}), wp::dot(basis{s}[5], u{s}));"""
    # Over a block (:func:`region_phase`): the left-out regions over every thread, the join voted by the block, the
    # walk made anew over the block's warps (a narrow world's by the first warp) and its levels by the block's warps
    # in turn (region_sweep.walk_levels' turns).
    sync, me, stride = ("__syncthreads();", "tid", "(int)blockDim.x") if block else ("SYNC();", "lane", "STRIDE")
    voted = "__syncthreads_or(joined != 0)" if block else "BALLOT(joined != 0) != 0u"
    parts = indent(WALK_PARTS, "    ")
    if block:
        # A world wider than the block's lanes is walked by every warp of the block, a part each; the first warp
        # walks a narrower one (its acting count to the block by the turns' second word). The block levels the walk.
        remake = f"""        const int s0 = left_span[0], order_end = world_order_end.data[world];
        const auto BLOCK_SYNC = [&]() {{ __syncthreads(); }};
{parts}
        const bool wide = order_end - s0 > (int)blockDim.x;
        int acting = wide ? walk_wide(s0, order_end, (int)blockDim.x >> 5, warp) : 0;
        if (!wide) {{
            if (warp == 0) {{
                acting = count_part(s0, order_end);
                place_part(s0, s0, order_end, 0, acting);
                if (lane == 0) s_level_turn[1] = acting;
            }}
            __syncthreads();
            acting = s_level_turn[1];
        }}
{walk_levels(turns="s_level_turn[0]")}
        __syncthreads();"""
    else:
        remake = f"""        SYNC();
        const int s0 = left_span[0], order_end = world_order_end.data[world];
        const auto BLOCK_SYNC = [&]() {{ SYNC(); }};
{parts}
        int acting = count_part(s0, order_end);
        place_part(s0, s0, order_end, 0, acting);
        SYNC();
{walk_levels()}
        SYNC();"""
    # The tests: each forms its two bodies' twists (the per-test form, every form's). A block (its lanes many, its
    # tests often many a lane) forms each body's twist once instead where that takes a lane fewer rounds.
    tests = f"""    for (int place = left_first + {me}; place < left_last; place += {stride}) {{
        const int g = reg_walk.data[place];
        const wp::vec2i sides = reg_body.data[g];
        if (reg_state.data[g] != {int(REGION_LEFT)}) continue;
{indent(twist.format(s="", f="sides[0]"), "        ")}
        wp::vec3 v_b, w_b;
        if (sides[1] >= 0) {{
{indent(twist.format(s="_b_", f="sides[1]"), "            ")}
            v_b = v_b_;
            w_b = w_b_;
        }}
        const float sign = (reg_kind.data[g] & {int(KIND_SECOND)}) != 0 ? -1.0f : 1.0f;
        const wp::mat33 frame = reg_frame.data[g];
        const wp::vec3 n(frame.data[0][0], frame.data[0][1], frame.data[0][2]);
        const wp::vec3 t0(frame.data[1][0], frame.data[1][1], frame.data[1][2]);
        const wp::vec3 t1(frame.data[2][0], frame.data[2][1], frame.data[2][2]);
        wp::vec3 sn = wp::mul(RegionLaw::frame_twist(v, w, reg_arm.data[g], n, t0, t1).v0, sign);
        if (sides[1] >= 0) {{
            sn = wp::sub(sn, wp::mul(RegionLaw::frame_twist(v_b, w_b, reg_arm_b.data[g], n, t0, t1).v0, sign));
        }}
        // Where its closing bound rules a closing out none closes (region_laws.stays_open), as in the phase.
        if (RegionLaw::stays_open(sn, reg_bound.data[g])) continue;
        const wp::vec2i span = reg_cand.data[g];
        bool closes = false;
        for (int i = span[0]; i < span[1]; ++i) {{
            const wp::vec4 record = cand.data[i];
            closes = closes || RegionLaw::candidate_closes(record, sn, cand_floor.data[i]);
        }}
        if (closes) {{
            reg_state.data[g] = {int(REGION_IDLE)};
            joined = 1;
        }}
    }}"""
    if block:
        tests = f"""    // The bodies' twists once a body (each formed once, then the tests {_VERIFY_TURNS} at a time) where that takes a lane
    // fewer rounds than each test forming its own: the same twists, the shorter wait.
    const int free_first = world_free_start.data[world], free_last = world_free_start.data[world + 1];
    const int test_rounds = (left_last - left_first + {stride} - 1) / ({stride});
    const int body_rounds = (free_last - free_first + {stride} - 1) / ({stride})
        + (left_last - left_first + {_VERIFY_TURNS} * {stride} - 1) / ({_VERIFY_TURNS} * {stride});
    if (test_rounds > body_rounds) {{
        {{
            // Each free body's twist as the sweep leaves it (its walk coordinates through its basis, as the phase forms
            // it), once a body: a test then reads its two bodies' twists, not their coordinates and bases.
            for (int k = free_first + {me}; k < free_last; k += {stride}) {{
                const int f = world_free.data[k];
    {indent(twist.format(s="", f="f"), "            ")}
                body_end_twist.data[f] = wp::spatial_vector(v, w);
            }}
            {sync}
        }}
        // The tests, {_VERIFY_TURNS} places of a lane at once: their words in flight together, then each test in turn.
        for (int base = left_first + {me}; base < left_last; base += {_VERIFY_TURNS} * {stride}) {{
            int g[{_VERIFY_TURNS}];
            wp::vec2i sides[{_VERIFY_TURNS}];
    #pragma unroll
            for (int t = 0; t < {_VERIFY_TURNS}; ++t) {{
                const int place = base + t * {stride};
                g[t] = place < left_last ? reg_walk.data[place] : -1;
            }}
    #pragma unroll
            for (int t = 0; t < {_VERIFY_TURNS}; ++t) sides[t] = g[t] >= 0 ? reg_body.data[g[t]] : wp::vec2i(-1, -1);
            int state[{_VERIFY_TURNS}], kind[{_VERIFY_TURNS}];
            wp::mat33 frame[{_VERIFY_TURNS}];
            wp::vec3 arm[{_VERIFY_TURNS}], arm_b[{_VERIFY_TURNS}];
            wp::vec2 bound[{_VERIFY_TURNS}];
            wp::spatial_vector twist[{_VERIFY_TURNS}], twist_b[{_VERIFY_TURNS}];
    #pragma unroll
            for (int t = 0; t < {_VERIFY_TURNS}; ++t) {{
                const int at = g[t] >= 0 ? g[t] : 0;
                state[t] = g[t] >= 0 ? reg_state.data[at] : {int(REGION_IDLE)};
                kind[t] = reg_kind.data[at];
                frame[t] = reg_frame.data[at];
                arm[t] = reg_arm.data[at];
                arm_b[t] = reg_arm_b.data[at];
                bound[t] = reg_bound.data[at];
                twist[t] = body_end_twist.data[sides[t][0] >= 0 ? sides[t][0] : 0];
                twist_b[t] = body_end_twist.data[sides[t][1] >= 0 ? sides[t][1] : 0];
            }}
    #pragma unroll
            for (int t = 0; t < {_VERIFY_TURNS}; ++t) {{
                if (state[t] != {int(REGION_LEFT)}) continue;
                const wp::vec3 v(twist[t][0], twist[t][1], twist[t][2]), w(twist[t][3], twist[t][4], twist[t][5]);
                const float sign = (kind[t] & {int(KIND_SECOND)}) != 0 ? -1.0f : 1.0f;
                const wp::vec3 n(frame[t].data[0][0], frame[t].data[0][1], frame[t].data[0][2]);
                const wp::vec3 t0(frame[t].data[1][0], frame[t].data[1][1], frame[t].data[1][2]);
                const wp::vec3 t1(frame[t].data[2][0], frame[t].data[2][1], frame[t].data[2][2]);
                wp::vec3 sn = wp::mul(RegionLaw::frame_twist(v, w, arm[t], n, t0, t1).v0, sign);
                if (sides[t][1] >= 0) {{
                    const wp::vec3 v_b(twist_b[t][0], twist_b[t][1], twist_b[t][2]);
                    const wp::vec3 w_b(twist_b[t][3], twist_b[t][4], twist_b[t][5]);
                    sn = wp::sub(sn, wp::mul(RegionLaw::frame_twist(v_b, w_b, arm_b[t], n, t0, t1).v0, sign));
                }}
                // Where its closing bound rules a closing out none closes (region_laws.stays_open), as in the phase.
                if (RegionLaw::stays_open(sn, bound[t])) continue;
                const wp::vec2i span = reg_cand.data[g[t]];
                bool closes = false;
                for (int i = span[0]; i < span[1]; ++i) {{
                    const wp::vec4 record = cand.data[i];
                    closes = closes || RegionLaw::candidate_closes(record, sn, cand_floor.data[i]);
                }}
                if (closes) {{
                    reg_state.data[g[t]] = {int(REGION_IDLE)};
                    joined = 1;
                }}
            }}
        }}
    }} else {{
{indent(tests, "    ")}
    }}"""
    text = f"""{{
    {sync}
    const wp::vec2i left_span = world_regions.data[world];
    // The left-out regions are the walk's places after its acting ones (the last level's end) up to the order's end
    // (region_sweep.level_acting); a verify that opens one makes the walk anew below.
    const int left_levels = world_levels.data[world];
    const int left_first = left_levels > 0 ? level_end.data[left_span[0] + left_levels - 1] : left_span[0];
    const int left_last = world_order_end.data[world];
    int joined = 0;
{tests}
    if ({voted}) {{
        iteration_changed = 1;
        if ({iteration} + 1 == {budget}) {budget} += 1;
        // The opened regions take their places in the order: the walk and its levels anew (region_sweep.level_acting's
        // statements), so the next sweep takes the order on every region that can act, as it stands.
{remake}
    }}
}}"""
    return indent(text, pad)


def world_lanes(support: int, *, free_rows: bool) -> int:
    """A world's lanes (switch 2): the power of two over its rows' widest ``support``, 8 at least, 16 with the free
    bodies' run (6 + 6 entries a row), up to a warp's 32: a lane holds at most one entry of a row."""
    lanes = 8 if support <= 8 else 16 if support <= 16 else 32
    return max(lanes, 16) if free_rows else lanes


def mf_gs_staging(
    max_constraints: int,
    mf_max_constraints: int,
    max_world_dofs: int,
    *,
    colored: bool = False,
    grouped: bool = False,
    regions: bool = False,
    flow: bool = False,
    places: RowPlaces = STAGED_ROWS,
    traces: bool = False,
    wide_rows: bool = False,
    free_rows: bool = True,
    active_rows: bool = False,
    coordinate_levels: bool = False,
    groups: int = 1,
    lanes: int = 32,
    split: bool = False,
) -> Staging:
    """The shared working set of a world's matrix-free sweep block (:func:`mf_gs_source` declares it), in order of
    reuse: the world's velocity, its dense rows' and the free bodies' run's (``free_rows``) impulses at ``places``, the
    dense rows' staged metadata, ``colored`` the colors' staged offsets (and its trace's warp maxima), ``active_rows``
    the active enumeration's bitsets, ``coordinate_levels`` the leveled units' marks, for ``groups`` worlds of ``lanes``
    lanes; ``grouped`` a block a world as ``colored``, its trace's warp maxima."""
    # A packed row's sides stage with its metadata (the classes' word takes the widths' place).
    sides = None if wide_rows else "widths"
    dense = WorldRows(
        max_constraints,
        places.metadata,
        stream=places.dense_impulses is Place.GLOBAL,
        sides=sides,
        split=split,
        head=places.dense_rows,
        floor=places.dense_floor,
    )
    free = WorldRows(
        mf_max_constraints,
        Place.STREAMED,
        free=True,
        stream=places.impulses is Place.GLOBAL or not free_rows,
        split=split,
        slots=places.free_slots,
        wide=places.wide_slots,
    )
    # The impulses each visit reads and writes, then the record words it only reads (streamed where declined).
    rows = sorted((*dense.buffers(), *free.buffers()), key=lambda buffer: buffer.alternative is not Place.GLOBAL)
    levels = ()
    if colored:
        # Each body's latest unit's turn, at its first coordinate (_colored_units); under regions the remake's turn
        # words (region_verify: the level count, a narrow walk's acting count).
        levels = (
            (Buffer("s_level_turn", "int", 2), *((Buffer("s_ran", "int", ran_slots(max_world_dofs)),) if flow else ()))
            if regions
            else (Buffer("s_ran", "int", ran_slots(max_world_dofs)),)
        )
        if traces:
            levels += block_argmax(COLORED_THREADS // 32, warp="warp", lane="lane")[1]
    elif grouped and traces:
        levels = block_argmax(COLORED_THREADS // 32, warp="warp", lane="lane")[1]
    # The world's velocity change, staged unless it lives in global memory (RowPlaces.velocity).
    velocity = (
        (
            Buffer("s_v", "float", max_world_dofs),
            *((Buffer("s_sweep_v", "float", max_world_dofs),) if wide_rows else ()),
        )
        if places.velocity is Place.SHARED
        else ()
    )
    buffers = (
        *velocity,
        *rows,
        *levels,
        *(active_words(dense.staged_rows) if active_rows else ()),
        *((Buffer("s_level_mark", "int", (dense.staged_rows + 31) // 32),) if coordinate_levels else ()),
    )
    if free.wide:
        # The wide head lies last, so the stage's launch takes a prefix of the wide launch's bytes
        # (memory.Staging.resized).
        buffers = (
            *(buffer for buffer in buffers if buffer.name != free.impulses),
            *(buffer for buffer in buffers if buffer.name == free.impulses),
        )
    return Staging(
        buffers,
        **({"threads": COLORED_THREADS} if colored or grouped else {"threads": lanes * groups}),
        **({"groups": groups, "layout": "arrays"} if groups > 1 else {}),
    )


_OWN_CYCLES_START = """    const unsigned int own_from = (unsigned int)clock();"""
_OWN_CYCLES_END = """        if (lane == 0) world_cost.data[world] = 32 - __clz((int)((unsigned int)clock() - own_from));
"""
"""A world's cost for the next call's order (:func:`.sweep_levels.claim_worlds`): the bit length of its own cycles,
from its start to its last sweep's end, stored at each sweep's end inside the sweep loop (before the loop's exit makes
a world wait for the warp-mate it shares lanes with), so only its start's 32-bit clock stays live across the loop. A lane group's body is CUDA's only (:func:`.device_preludes.group_native`); the CPU's
one thread keeps no cost: its worlds run one at a time."""


GROUP_WARPS = sweep_units.GROUP_WARPS
"""The warps of a grouped world's block (:func:`group_rows`), its :data:`COLORED_THREADS`: a group each at once."""
if COLORED_THREADS != 32 * GROUP_WARPS:
    raise ValueError("a grouped world's block holds the colored mapping's threads, a warp a group")


# A world's group of blocks meets at a counter (its job word 6, from zero each launch): every block counts itself in,
# after its own stores, and waits until the group's every block has; with ``voted`` each block's loud change joins the
# world's (job word 5) as it counts in.
GROUP_BARRIER = """        {{
            const bool voted = {vote};
            ++wide_meets;
            if (tid == 0) {{
                if (voted) atomicOr(wide_job + 5, 1);
                __threadfence();
                atomicAdd(wide_job + 6, 1);
                const volatile int* const job = wide_job;
                while (job[6] < wide_meets * group) {{ }}
                __threadfence();
            }}
            __syncthreads();
        }}
"""


def group_rows(loop: Callable[[str, str, str], str], pad: str, *, group: bool = False) -> str:
    """A grouped world's dense rows (:class:`.row_slots.RowGroups`) by its block: the rows before its first group (the
    topology's: limits, links, mimics) in order on the first warp, then its colors in turn, a color's groups a warp
    each, its rows in order (``loop(begin, end, pad)``: the rows ``[begin, end)``), a barrier after each. A color's
    groups share no tree, so they write no coordinate another reads: the colors' order is the sweep's.

    Over a world's ``group`` of blocks (:func:`worlds`' ranks) the world's block (rank 0) runs the topology rows, then
    opens the sweep (its job words, :data:`WIDE_JOB`), which the other blocks take, the world's end last. Every block
    reads the colors (the group's tables): a color with more groups than a block's warps is the group's, its groups
    spread over the blocks' warps (``warp * group + rank``), a narrow one the world's block's. The blocks meet
    (:data:`GROUP_BARRIER`) after a color of the group and before one, the sweep's last with their exit votes, which
    the world's block takes; between two narrow colors its own barrier suffices. The same visits in the same colors."""
    inner = "        "
    topology = f"""    if (warp == 0{" && rank == 0" if group else ""}) {{
        const int row_begin = 0;
        const int row_end = min(group_rows[group_offsets[0]], m_dense);
{loop("row_begin", "row_end", inner)}
    }}
    __syncthreads();"""
    run = loop("row_begin", "row_end", inner + "    ")

    def visit(at: str, step: str) -> str:
        return f"""        for (int row_group = {at}; row_group < color_end; row_group += {step}) {{
            const int row_begin = min(group_rows[row_group], m_dense);
            const int row_end = min(group_rows[row_group + 1], m_dense);
{run}
        }}"""

    if not group:
        colors = f"""    for (int color = 0; color < group_colors; ++color) {{
        const int color_first = group_offsets[color], color_end = group_offsets[color + 1];
{visit("color_first + warp", str(GROUP_WARPS))}
        __syncthreads();
    }}"""
        return indent(f"{{\n{topology}\n{colors}\n}}", pad)
    wide = f"group > 1 && group_offsets[{{c}} + 1] - group_offsets[{{c}}] > {GROUP_WARPS}"
    meet = GROUP_BARRIER.format(vote="__syncthreads_or(iteration_changed != 0)")
    colors = f"""    for (;;) {{
    // The world's block opens the sweep (its topology rows ran); the others take it, or the world's end.
    if (group > 1) {{
        ++wide_gen;
        if (rank == 0) {{
            if (tid == 0) {{
                __threadfence();
                atomicExch(wide_job, wide_gen);
            }}
        }} else {{
            if (tid == 0) {{
                const volatile int* const job = wide_job;
                while (job[0] < wide_gen) {{ }}
                __threadfence();
            }}
            __syncthreads();
            const volatile int* const job = wide_job;
            if (job[3] != 0) break;
        }}
    }}
    for (int color = 0; color < group_colors; ++color) {{
        const int color_first = group_offsets[color], color_end = group_offsets[color + 1];
        const bool wide = {wide.format(c="color")};
        if (wide || rank == 0) {{
            const int place_at = wide ? warp * group + rank : warp;
            const int place_step = wide ? {GROUP_WARPS} * group : {GROUP_WARPS};
{visit("color_first + place_at", "place_step")}
        }}
        if (group > 1 && (wide || color + 1 == group_colors || ({wide.format(c="color + 1")}))) {{
{indent(meet, "    ")}
        }} else {{
            __syncthreads();
        }}
    }}
    // The world's block takes the group's votes (each counted in before the sweep's last meeting) and clears them for
    // the next sweep, which it opens; the others reset theirs and take its next.
    if (group > 1) {{
        if (rank == 0) {{
            if (tid == 0) {{
                volatile int* const job = wide_job;
                if (job[5] != 0) {{
                    job[5] = 0;
                    iteration_changed = 1;
                }}
            }}
        }} else {{
            iteration_changed = 0;
        }}
    }}
    if (rank == 0) break;
    }}"""
    return indent(f"{{\n{topology}\n{colors}\n}}", pad)


def mf_gs_source(sweep: GeneralSweep) -> tuple[str, Staging]:
    """The native body of the matrix-free Gauss-Seidel sweep and its staging: a world's dense rows
    (:func:`packed_operator`) and free-body rows (:func:`free_body_rows`), then its end-of-sweep passes, on the world's
    velocity change in walk coordinates (:class:`FreeBodyCoordinates`), from zero or ``seeded``
    (:func:`.sweep.impulse_seed`), added to the walk's ``du`` for the step's one leave. Every switch is the
    configuration's (:class:`GeneralSweep`)."""
    places, solve_policy = sweep.places, sweep.solve_policy
    contact_torsion, seeded, wide_rows = sweep.torsion, sweep.seeded, sweep.wide_rows
    free_rows, max_row_dofs = sweep.free_rows, sweep.max_row_dofs
    colored, grouped = sweep.colored, sweep.grouped
    # A world a block: the colored mapping's, or the dense line by colors of groups (group_rows).
    block = colored or grouped
    lanes = sweep.lanes
    M_D, M_MF, D = sweep.max_constraints, sweep.mf_max_constraints, sweep.max_world_dofs
    row_lanes = Lanes(lanes)
    laws = sweep.laws
    # A packed row's coordinates decode from its staged sides, as a streamed record's do.
    sides = None if wide_rows else "classes" if sweep.support_classes else "widths"
    split = sweep.split
    dense = WorldRows(
        M_D,
        places.metadata,
        stream=places.dense_impulses is Place.GLOBAL,
        lanes=row_lanes,
        sides=sides,
        split=split,
        head=places.dense_rows,
        floor=places.dense_floor,
    )
    free = WorldRows(
        M_MF,
        Place.STREAMED,
        free=True,
        stream=places.impulses is Place.GLOBAL,
        lanes=row_lanes,
        split=split,
        slots=places.free_slots,
        wide=places.wide_slots,
    )
    dense_op = packed_operator(
        max_row_dofs,
        row_lanes,
        wide=wide_rows,
        rows=dense,
        classes=sweep.support_classes,
        held=places.velocity is not Place.SHARED,
    )
    free_op = free_body_rows(row_lanes, max_row_dofs)
    staging = sweep.staging()

    dense_count = "int m_dense = world_constraint_count.data[world];"
    dense_loads = indent(dense.loads("i"), "        ")
    mf_load = free.loads("i")
    mf_store = free.stores("i")

    budget, left_verify = "iterations", ""

    # Where spin rows may appear, every friction contact's record (its Gram couplings and metric) is formed once per
    # launch before the sweeps (sweep.friction_record_stage), the dense rows' and the free bodies', and its visits only
    # reduce velocity terms and project; the colored units read their contacts' records whatever the laws. A contact
    # holds three rows at least: a slot each.
    metric = dense_record = free_record = None
    friction_laws = metric_stage = ""
    # A free row's coordinates are its sides', from its header.
    sides = free.visit("normal") + "\n" + free.unpack_header("")
    if contact_torsion:
        metric = "FRICTION"

        def dense_record(row: str) -> str:
            return friction_metric(row, rows=3)

        friction_laws = formed_friction_laws(metric, "    ") + "\n"
        # The dense contacts' records: formed here, or by the stage launch that runs this text without its sweeps
        # ahead of it (``formed``), so the sweep's own code holds none of the formation.
        if sweep.formed != "ahead":
            # A block form's stage launch: the world's group of blocks, a contact a thread (spread).
            spread = block and sweep.formed == "stage"
            metric_stage = (
                friction_record_stage(
                    dense,
                    dense_op,
                    end="m_dense",
                    record=dense_record,
                    block=grouped or spread,
                    spread=spread,
                    pad="    ",
                )
                + "\n"
            )
        if free_rows and not colored and not sweep.regions and sweep.formed != "stage":

            def free_record(row: str) -> str:
                # The free bodies' rows follow the dense ones in the line, from its body base.
                return friction_metric(row, "free_row0", rows=3)

            stage = friction_record_stage(
                free, free_op, end="mf_contact_end", record=free_record, declare=sides, pad="    "
            )
            metric_stage += stage + "\n"

    if block:
        # The block's vote: every warp's rows and units (a barrier); the first thread records the solve.
        quiet = "!__syncthreads_or(iteration_changed != 0)"
        # Under regions a join where the budget ends takes a sweep more (region_verify).
        budget = "sweep_budget" if sweep.regions else "iterations"
        sweep_end = sweep_exit(
            solve_policy,
            quiet=quiet,
            iteration=ITERATION,
            iterations=budget,
            lane="tid",
            argmax="BLOCK_ARGMAX",
        )
        head = "    const int lane = LANE;\n    const int tid = (int)threadIdx.x;\n    const int warp = tid >> 5;"
        first_warp, end_first_warp, barrier = "    if (warp == 0) {\n", "    }\n", "__syncthreads();"
    world_end = leave = ""
    if colored and sweep.regions:
        # The free bodies' contacts are regions: the world's open ones by their levels over the block, the sweeps' end
        # tests the left-out ones over it.
        free_phase = region_phase("    ", block=True, group=sweep.world_blocks)
        left_verify = region_verify(budget=budget, block=True) + "\n"
        trace_reduction = ""
        if solve_policy.traces:
            trace_reduction = block_argmax(COLORED_THREADS // 32, warp="warp", lane="lane")[0]
        offsets = metric_build = ""
        if not sweep.world_blocks:
            # The flow's marks (_region_flow): no region ran yet (turns are at least zero), before the sweeps' barrier.
            metric_build = (
                f"    for (int d = tid; d < {ran_slots(D)}; d += blockDim.x) s_ran[d] = -1;\n    int region_turn = 0;\n"
            )
    elif colored:
        # The world's entries of the pool: the stride of its units' staged fields and words in its block
        # (sweep_units.SLOT_FIELDS).
        units = "unit_entries"
        share = sweep_units.unit_capacity(M_MF)

        def unit_metric(row: str) -> Callable[[str], str]:
            # A unit's record in its world's line: by its place in the colors' order, field-major at the share's stride
            # from the free rows' first record (every dense contact's normal precedes the body base), where the world's
            # units fit its share: its visit's loads do not wait for its color slot, and a color's units read each float
            # together. A world past its share keeps them where the warp sweep keeps its free rows' records, by its
            # normal row (a friction unit's rows are three apart at least), each record's floats together: the line
            # holds a record for every three rows either way, so the records take no memory of their own.
            first = f"(free_row0 / 3) * {FRICTION_METRIC_FLOATS}"
            return lambda place: (
                f"(unit_record_line + (records_by_row ? ((free_row0 + ({row})) / 3) * {FRICTION_METRIC_FLOATS} "
                f": {first} + ({place})))"
            )

        free_phase = _colored_units(free, laws, M_MF, units, unit_metric("unit_row"), "record_stride")
        # The units' records at the line's stride, with the normal's couplings a unit moves its rows' terms by.
        listed = ("color_offset.data[off_color + color_count.data[world]]", COLOR_SLOT, "record_stride")
        # No unit ran yet (turns are at least zero), before the records' barrier.
        ran = f"    for (int d = tid; d < {ran_slots(D)}; d += blockDim.x) s_ran[d] = -1;\n"
        unit_records = friction_record_stage(
            free,
            free_op,
            end="mf_contact_end",
            record=unit_metric("unit.x"),
            declare=sides,
            units=listed,
            spread=sweep.formed == "stage",
            pad="    ",
        )
        # The units' records: formed here (inline), by the stage launch over the world's group of blocks (stage), whose
        # sweep forms none (ahead).
        metric_build = {"inline": ran + unit_records, "stage": unit_records, "ahead": ran}[sweep.formed]
        trace_reduction = ""
        if solve_policy.traces:
            trace_reduction = block_argmax(COLORED_THREADS // 32, warp="warp", lane="lane")[0]
        # The world's storage slot (a colored world has one): its first entry of the pool holds its colors' offsets and
        # its staged units.
        offsets = (
            "\n    const int unit_slot = world_slot.data[world];"
            "\n    const int off_unit = slot_first.data[unit_slot];"
            "\n    const int off_color = off_unit;"
            "\n    const int unit_entries = slot_need.data[unit_slot];"
            # Its units' records in its line: by place where its units fit its share, else by normal row (unit_metric).
            "\n    const bool records_by_row = color_offset.data[off_color + color_count.data[world]] > "
            f"{share};"
            f"\n    const int record_stride = records_by_row ? 1 : {share};"
            "\n    float* const unit_record_line = friction_metric.data + static_cast<size_t>(line) * "
            f"{line_size('friction_metric')};"
        )
    elif grouped:
        # The world's groups (sweep_routes.group_rows): its colors' first groups and its groups' first rows.
        free_phase = ""
        metric_build = ""
        trace_reduction = ""
        if solve_policy.traces:
            trace_reduction = block_argmax(COLORED_THREADS // 32, warp="warp", lane="lane")[0]
        offsets = (
            "\n    const int group_slot = row_group_slot.data[world];"
            "\n    const int group_colors = row_group_colors.data[group_slot];"
            "\n    const int* const group_offsets = row_group_offsets.data + "
            "(size_t)group_slot * row_group_offsets.shape[1];"
            "\n    const int* const group_rows = row_group_rows.data + (size_t)group_slot * row_group_rows.shape[1];"
        )
    else:
        violation = active_violation(dense, dense_op, count="m_dense") if sweep.active_rows else None
        # The world's own cycles (its cost class, sweep_levels.COST_CLASSES): read at each sweep's end inside the
        # loop, where the world's lanes have not yet waited for a warp-mate at the loop's exit.
        owns = lanes > 1
        # Under regions a join where the budget ends takes a sweep more (region_verify).
        budget = "sweep_budget" if sweep.regions else "iterations"
        sweep_end = (_OWN_CYCLES_END if owns else "") + end_sweep(
            solve_policy, iteration=ITERATION, violation=violation, iterations=budget
        )
        head = "    const int lane = LANE;" + ("\n" + _OWN_CYCLES_START if owns else "")
        first_warp = end_first_warp = ""
        barrier = "SYNC();"
        free_phase = ""
        if sweep.regions:
            # The free bodies' contacts are regions: the world's open ones, by the one visit; the sweeps' end tests
            # the left-out ones.
            free_phase = region_phase("    ")
            left_verify = region_verify(budget=budget) + "\n"
        elif free_rows:
            free_phase = rows_block(free, free_op, laws, metric=metric, formed=free_record, pad="    ")
        offsets = trace_reduction = metric_build = ""

    # Before the sweeps and after them the world's block (rank 0) works alone: the others serve its stages.
    alone, end_alone = "", ""
    if sweep.world_blocks:
        # A world's group of blocks (group_stage): its block (rank 0) runs every phase, the others serve its stages
        # wider than the block and leave the sweeps at the world's end, which the block posts after its last sweep.
        head += f"\n    int wide_gen = 0, wide_meets = 0;\n    int* const wide_job = wide_jobs.data + (size_t)world * {WIDE_JOB};"
        first_warp = "    if (warp == 0 && rank == 0) {\n"
        alone, end_alone = "    if (rank == 0) {\n", "    }\n"
        leave = "        if (rank != 0) break;\n"
        world_end = """
    if (rank == 0 && group > 1) {
        ++wide_gen;
        if (tid == 0) {
            volatile int* const job = wide_job;
            job[3] = 1;
            __threadfence();
            atomicExch(wide_job, wide_gen);
        }
    }"""
    thread, threads = ("tid", "(int)blockDim.x") if block else ("lane", "STRIDE")
    seed = ""
    if seeded:

        def dense_seed(begin: str, end: str, pad: str) -> str:
            return impulse_seed(
                end,
                dense.peek("i"),
                dense_op.axpy((("row", "seed"),), "        ", row="i"),
                row_lanes.sync(),
                declare=tuple(line for line in dense.visit("i").split("\n") if line),
                begin=begin,
                pad=pad,
            )

        seeds = (group_rows(dense_seed, "    ") if grouped else dense_seed("0", "m_dense", "    "),)
        seeds += (
            (
                impulse_seed(
                    "m_mf",
                    free.load("i"),
                    free_op.axpy((("row", "seed"),), "        ", row="i"),
                    row_lanes.sync(),
                    declare=(
                        *free.visit("i").split("\n"),
                        *(
                            ("const int dof_a = rec_a.x, dof_b = rec_b.x;",)
                            if split
                            else (
                                "const int packed_dofs = rec_a.x;",
                                "const int dof_a = packed_dofs >> 16, dof_b = (packed_dofs << 16) >> 16;",
                            )
                        ),
                    ),
                ),
            )
            if free_rows and not sweep.regions
            else ()
        )
        seed = (
            f"{alone}{seeds[0]}\n{end_alone}"
            if grouped
            else f"{first_warp}{chr(10).join(seeds)}\n{end_first_warp}    {barrier}\n"
        )
    # Wide rows, which the bilateral block corrected, can be collinear (a jammed loop), and collinear rows can trade
    # impulses forever with no effect on the velocity, each loud alone. The velocity is the solve's answer, so a wide
    # sweep also ends when no coordinate moved past the exit law's floor of the largest one: factor coordinates share
    # the kinetic energy's metric, and a coordinate the rows cancel to zero keeps their rounding, far above its own size.
    sweep_vote = ""
    if wide_rows:
        metric_build += f"""
    for (int d = {thread}; d < {D}; d += {threads}) s_sweep_v[d] = s_v[d];"""
        sweep_vote = f"""        {barrier}
        int velocity_moved = 0;
        float velocity_floor = 0.0f;
        for (int d = 0; d < {D}; ++d) velocity_floor = fmaxf(velocity_floor, fabsf(s_v[d]));
        velocity_floor *= {QUIET_LITERAL};
        for (int d = {thread}; d < {D}; d += {threads}) {{
            const float now = s_v[d];
            if (!(fabsf(now - s_sweep_v[d]) <= velocity_floor)) velocity_moved = 1;
            s_sweep_v[d] = now;
        }}
        if ({"!__syncthreads_or(velocity_moved != 0)" if colored else "BALLOT(velocity_moved != 0) == 0u"}) iteration_changed = 0;
"""
    admission = ""
    triples = ""
    if contact_torsion and sweep.one_tree and not block:
        # Every world holds one tree: K1's order. The row loop solves every contact's normal and every other row, then
        # the friction triples follow in the shared phase (sweep.triple_phase), each on its contact's record.
        spin = dense.spin_of(dense.parent("i", enclosed=True), "m_dense")
        admission = f"if (row_type == {ROW_FRICTION} && {spin} >= 0) continue;"
        record = ", ".join(f"{name} = friction_record[{k}]" for k, name in enumerate(FORMED_RECORD))
        triples = triple_phase(
            dense,
            dense_op,
            laws,
            count="m_dense",
            record=f"const float4* const friction_record = reinterpret_cast<const float4*>({dense_record('head')});\n"
            f"const float4 {record};",
            law=f"{metric}({', '.join(FORMED_RECORD)}, ",
            heads=lambda row: f"{dense.spin_of(row, 'm_dense')} >= 0 && {dense.kind(row)} == {ROW_CONTACT}",
            pad="        ",
        )

    def unit_text(rows: WorldRows, op: SparseFactor) -> str:
        return dense_units(rows, op, laws, active=sweep.active_rows, record=dense_record, pad="")

    units = unit_text(dense, dense_op) if sweep.contact_units else ""
    level_marks = level_phase = skip = ""
    if sweep.coordinate_levels:
        level_marks, level_phase, skip = coordinate_levels(
            dense, dense_op, laws, metric=metric, formed=dense_record, triple_phase=bool(triples)
        )
        units = f"{skip}\n{units}" if units else skip
    # The warp mapping's dense rows step to the rows that can act (sweep.LiveOrder) where their applies are the only
    # ones inside their loop.
    live = None
    if dense.owned and not (units or sweep.active_rows or block):
        live = (dense_op.velocity, str(D))
    dense_phase = rows_block(
        dense,
        dense_op,
        laws,
        admission=admission,
        metric=metric,
        formed=dense_record,
        selected=sweep.active_rows,
        unit=units,
        triple_phase=bool(triples),
        live=live,
        pad="        ",
    )
    if grouped:
        # A run of the world's rows: the topology rows on the first warp, then each group's on its warp. Where the
        # velocity lies in global memory and the records stream, a run's visits carry the velocity they kept
        # (Velocity.carried) and load the next visit's operands while a visit runs (RUN_AHEAD).
        run_rows, run_op = dense, dense_op
        if dense_op.keeps_velocity and dense_op.key is not None:
            run_rows = dataclasses.replace(dense, ahead=RUN_AHEAD_WORDS, kept_stores=RUN_KEPT_STORES)
            velocity = dataclasses.replace(dense_op.velocity, carried=True)
            run_op = dataclasses.replace(dense_op, velocity=velocity, ahead=RUN_AHEAD_ENTRIES)
            if sweep.contact_units:
                units = "\n".join(text for text in (skip, unit_text(run_rows, run_op)) if text)
        dense_phase = group_rows(
            lambda begin, end, pad: rows_block(
                run_rows,
                run_op,
                laws,
                metric=metric,
                formed=dense_record,
                unit=units,
                end=end,
                begin=begin,
                pad=pad,
            ),
            "        ",
            group=sweep.world_blocks,
        )
    if units and sweep.contact_units:
        dense_phase = f"        int consumed_tangent = -1;\n{dense_phase}"
    if sweep.active_rows:
        # The active enumeration: each sweep's scan selects the dense rows it visits, after the last sweep's stores.
        # A triple left in the loop (two trees a world) is selected while its spin row holds impulse too.
        spin = (lambda parent: dense.spin_of(parent, "m_dense")) if contact_torsion and not triples else None
        scan = active_scan(
            dense, dense_op, count="m_dense", seeded=seeded, linked=True, torsion=contact_torsion, spin=spin, pad=""
        )
        dense_phase = f"        SYNC();\n{indent(scan, '        ')}\n{dense_phase}"
        metric_build += "\n" + active_selection(dense, count="m_dense", linked=True, pad="    ")
    dense_sweep = (
        dense_phase
        if grouped
        else f"{first_warp}{level_phase}{chr(10) if level_phase else ''}{dense_phase}\n{end_first_warp}"
        + ("        __syncthreads();" if colored else "")
    )
    sweeps = level_loop(
        (
            dense_sweep,
            triples,
            indent(free_phase, "    "),
        ),
        exit=leave + sweep_vote + left_verify + sweep_end,
        budget=budget,
    )
    # The streamed rows' records (row_record, row_metadata.record_layout). The row store's line (:mod:`.row_records`):
    # the dense rows, then the free bodies' from the line's body base (free_row0), at the store's line length; the
    # free rows' records follow the dense rows' where those stream.
    free_records = f"free_row0 * {record_words(False, split)}" if sweep.record_rows == 0 else "0"
    records = f"    const int off_rec_mf = line * {line_size('row_record')} + {free_records};"
    if dense.record:
        records = f"    const int off_rec_dense = line * {line_size('row_record')};\n" + records
    # The way back after the sweeps: the change in the walk's coordinates to the step's one leave (WalkTrees.du,
    # factor.walk.walk_leave_kernel).
    # The free bodies' run's rows (free_rows): their count, impulses' staging and publication.
    mf_count, free_view, free_loads, free_stores = "int m_mf = 0;", "", "", ""
    if free_rows:
        mf_count = "int m_mf = mf_constraint_count.data[world];"
        free_view = free.declaration()
        # Under regions the run's contact rows are the regions' candidates, whose impulses their visits publish: the
        # run stages none.
        staged_first = f"mf_contact_end + {thread}" if sweep.regions else thread
        free_loads = f"""
    for (int i = {staged_first}; i < m_mf; i += {threads}) {{
        {mf_load}
    }}"""
        free_stores = f"""
    for (int i = {staged_first}; i < m_mf; i += {threads}) {{
        {mf_store}
    }}"""
    dense_view = dense.declaration() + "\n" if dense.stream else ""
    # A world past a block's reach keeps its velocity change in its line of global memory (RowPlaces.velocity), read
    # and written as the stage's.
    velocity_view = ""
    if places.velocity is not Place.SHARED:
        velocity_view = "\n    float* s_v = sweep_velocity.data + (size_t)world * sweep_velocity.shape[1];"
        if wide_rows:
            velocity_view += f"\n    float* s_sweep_v = s_v + {D};"
    # The sweeps and the way back; the stage launch (``formed == "stage"``) forms the dense records and leaves.
    budget_line = f"    int {budget} = iterations;\n" if budget != "iterations" else ""
    add_back = f"""    for (int d = {thread}; d < {D}; d += {threads}) {{
        int global_dof = world_dof_indices.data[dof_map_base + d];
        if (global_dof >= 0) walk.du.data[global_dof] += s_v[d];
    }}
"""
    stores = f"""    for (int i = {thread}; i < m_dense; i += {threads}) {{
        {dense.stores("i")}
    }}{free_stores}
"""
    # The way back, a coordinate and a row a thread: a world a block's (its first block's, of a group), the warp
    # mapping's lanes.
    way_back = f"{alone}{add_back}{stores}{end_alone}"
    rest = f"""
{budget_line}{sweeps}{world_end}

    {barrier}
{way_back}"""
    if sweep.formed == "stage":
        seed, rest = "", ""
    # The rows' staging and the coordinates' clearing, a row and a coordinate a thread: a world a block's (its first
    # block's, of a group), the warp mapping's lanes.
    start = f"""{alone}    for (int i = {thread}; i < m_dense; i += {threads}) {{
{dense_loads}
    }}{free_loads}
    for (int d = {thread}; d < {D}; d += {threads}) s_v[d] = 0.0f;
{end_alone}"""
    records_stage = f"{alone + metric_stage + end_alone if grouped else first_warp + metric_stage + end_first_warp}"
    if block and sweep.formed == "stage":
        # A block form's stage launch (spread over the world's group of blocks): every block stages the rows' records
        # its share reads, then forms its contacts' records; it clears and solves nothing.
        start = f"""    for (int i = {thread}; i < m_dense; i += {threads}) {{
{dense_loads}
    }}
"""
        records_stage = metric_stage
    body = f"""{head}

    {dense_count}
    {mf_count}
    if (m_dense == 0 && m_mf == 0) return;
    if (m_dense > {M_D}) m_dense = {M_D};
    if (m_mf > {M_MF}) m_mf = {M_MF};
{dense.admission()}{free.admission()}    int mf_contact_end = mf_contact_rows_end.data[world];
    if (mf_contact_end > m_mf) mf_contact_end = m_mf;

    int dof_map_base = world * {D};
    const int line = dense_first.data[world];
    __builtin_assume(line >= 0);
    int off_dense = line * {line_size("world_impulses")};
    const int free_row0 = body_base.data[line];
    int off_mf = off_dense + free_row0;
{records}
    const size_t packed_base = (size_t)line * {line_size("packed_Z")};
    const size_t free_base = packed_base + (size_t)free_row0 * {int(max_row_dofs)};{offsets}

{staging.declarations("    ")}{velocity_view}
{dense_view}{free_view}{trace_reduction}
{friction_laws}
{start}    {barrier}
{seed}{records_stage}{metric_build}{chr(10) if level_marks else ""}{level_marks}
{rest}"""
    # The rows' layout is the configuration's: rows as wide as the world (and no classes) sit at their coordinates.
    packed = PACKED_SIDES_DENSE if not sweep.support_classes and max_row_dofs == D else PACKED_SIDES_BY_SIDE
    units_ = colored and not sweep.regions
    # The regions' remake of the walk's levels reads the lane group's words (MATCH, region_verify).
    sides = (
        packed,
        *((CLASS_SIDES,) if sweep.support_classes else ()),
        *((memory_words(_SLOTS),) if units_ else ()),
        *((LANE_WORDS,) if sweep.regions else ()),
    )
    records = bool(contact_torsion or units_)
    families = ((REGION_LAW,) if sweep.regions else ()) + ((IDLE_LAW,) if live is not None else ())
    return sweep.native(body, lanes=lanes, rows=sides, records=records, laws=families), staging


@dataclasses.dataclass(frozen=True)
class GeneralSweep(SweepConfig):
    """The general (matrix-free Gauss-Seidel) sweep's configuration: a world's dense rows (at most
    ``max_constraints``, packed whitened rows of ``max_row_dofs`` entries, by default as many as the world's
    ``max_world_dofs`` coordinates) and its free-body rows (at most ``mf_max_constraints``) on the world's velocity
    change in walk coordinates (:func:`mf_gs_source`).

    Its switches, each the plan's from the class's shape, the placement and the laws (:meth:`.sweep_plan.Sweep.build`):

    - the free bodies' level law (``levels``): sequential, or exact (colored: a block a world, a color's units at once,
      the worlds :class:`.sweep_units.ColoredRule` picks; where the free bodies' contacts are ``regions``, a level's
      regions at once, :func:`region_phase`);
    - the row record: ``places`` (:class:`RowPlaces`, the plan's storage choice): the dense rows' metadata staged or
      their records streamed; the free rows' always stream; all at the placement's layout (``records``, its first row);
    - the enumeration: ``active_rows`` (:func:`.sweep.active_scan`), else slot order (``coordinate_levels`` first);
      the friction triples after the rows (:func:`.sweep.triple_phase`, K1's order) where each world holds ``one_tree``;
    - the support class: ``max_row_dofs`` (the placement's packed support), ``support_classes`` (the rows' sides by
      their classes, :func:`.packed_rows.row_classes`), ``wide_rows`` (the bilateral block's), and the free bodies' 6 +
      6 class: ``free_rows`` their own run, else rows of the line;
    - ``seeded``: the seed.

    The warp mapping: ``lanes_per_world`` lanes a world in blocks of ``block_warps`` warps (one thread on the CPU); the
    colored one a block of :data:`COLORED_THREADS`; the ``grouped`` one (a world of many trees whose reservation placed
    its dense line by colors of groups, :class:`.row_slots.RowGroups`) a block of as many threads, a color's groups a
    warp each (:func:`group_rows`). Each walks its claim's bucket (:func:`worlds`).
    """

    max_constraints: int
    mf_max_constraints: int
    max_world_dofs: int
    device_arch: str
    _: dataclasses.KW_ONLY
    max_row_dofs: int | None = None
    places: RowPlaces = STAGED_ROWS
    seeded: bool = False
    wide_rows: bool = False
    support_classes: bool = False
    contact_units: bool = False
    coordinate_levels: bool = False
    lanes_per_world: int = 32
    block_warps: int = 1
    free_rows: bool = True
    one_tree: bool = False
    records: int | None = None
    split: bool = False
    regions: bool = False
    formed: str = "inline"
    world_blocks: bool = False
    grouped: bool = False
    line_stage: int = 0
    """The stage bytes its launch bound (the register line) is shaped at: a dense head's launch keeps its mapping's,
    the capacity's stage (:meth:`.sweep_plan.Sweep._dense_heads`), so a head moves no register and no arithmetic;
    ``0``: its own stage."""

    def __post_init__(self):
        if self.regions and not self.free_rows:
            raise ValueError("regions are the free bodies' run's units")
        if self.world_blocks and not (
            self.places.velocity is not Place.SHARED
            and (
                (self.colored and self.regions)
                or (
                    self.grouped
                    and self.places.dense_impulses is Place.GLOBAL
                    and self.places.metadata is Place.STREAMED
                )
            )
        ):
            raise ValueError(
                "a world's blocks serve its region levels or its colors of groups past a block, every operand they "
                "write or read in global memory"
            )
        if self.places.metadata is Place.STREAMED and self.record_rows:
            raise ValueError("a sweep that streams its dense rows' records reads them from the line's first row")
        if self.levels.law not in (SEQUENTIAL, EXACT):
            raise ValueError("the general sweep's free bodies step in order or by exact levels (colored)")
        if self.colored and not self.free_rows:
            raise ValueError("the colored mapping sweeps the free bodies' run: it needs free_rows")
        if self.split and self.colored and not self.regions:
            raise ValueError("the colored mapping's unit builder reads packed records, not split ones")
        if self.material and not self.weighted:
            raise ValueError("material contacts step by their weights: their sweeps are weighted")
        if self.formed not in ("inline", "ahead", "stage"):
            raise ValueError(
                f"a sweep's dense friction records are formed inline, ahead or as the stage, got {self.formed}"
            )
        if self.formed != "inline" and not (self.torsion or self.colored):
            raise ValueError(
                "friction records form ahead of a sweep only where it reads them: spin rows, colored units"
            )
        if self.formed == "stage" and (self.colored or self.grouped) and self.world_blocks:
            raise ValueError("a block form's records stage spreads over the world's group of blocks itself (spread)")
        if self.formed == "stage" and self.solve_policy.counts:
            raise ValueError("the records stage runs no sweep: it is no solve and carries no readout")
        if self.colored and (self.active_rows or self.contact_units or self.block_warps != 1):
            raise ValueError(
                "the colored mapping visits every dense row in turn, a world a block: its exit votes over it"
            )
        if self.grouped and (
            self.colored
            or self.free_rows
            or self.active_rows
            or self.coordinate_levels
            or self.one_tree
            or self.block_warps != 1
            or self.lanes_per_world != 32
        ):
            raise ValueError(
                "the grouped mapping sweeps the dense line of a world of many trees, a world a block, its colors' "
                "groups a warp each in slot order"
            )
        if self.coordinate_levels and (self.colored or self.active_rows or not self.support_classes):
            raise ValueError("the coordinate levels read the rows' support classes and precede every row in order")
        if self.block_warps not in (1, BLOCK_WARPS):
            raise ValueError(f"the warp mapping's block holds 1 or {BLOCK_WARPS} warps, got {self.block_warps}")
        if self.lanes_per_world not in (8, 16, 32) or (self.colored and self.lanes_per_world != 32):
            raise ValueError(f"the warp mapping takes 8, 16 or 32 lanes a world, got {self.lanes_per_world}")
        if self.free_rows and self.lanes_per_world < 16:
            raise ValueError("a free row's 6 + 6 entries take a lane each: the free bodies' run needs 16 lanes")
        dofs = self.max_world_dofs if self.max_row_dofs is None else self.max_row_dofs
        object.__setattr__(self, "max_row_dofs", int(dofs))

    @property
    def colored(self) -> bool:
        return self.levels.law == EXACT

    @property
    def record_rows(self) -> int:
        # The line's first row with a record: the placement's allocation, or without one the places' own.
        return record_first(self.max_constraints, self.places) if self.records is None else int(self.records)

    @property
    def lanes(self) -> int:
        """A world's lanes: the colored mapping's block of warps, the warp mapping's ``lanes_per_world`` on CUDA and one
        thread on the CPU (:func:`.execution.device_lanes`), which runs the world on either device."""
        return 32 if self.colored or self.grouped else device_lanes(self.device_arch, self.lanes_per_world)

    @property
    def groups(self) -> int:
        """Worlds a block on the device: the warp mapping's ``block_warps`` warps of worlds of ``lanes`` lanes on CUDA,
        one on the CPU (:func:`.execution.device_lanes`)."""
        return device_lanes(self.device_arch, 32 * self.block_warps) // self.lanes

    @property
    def spread(self) -> bool:
        """Whether the launch is a block form's records stage, spread over each world's group of blocks (:func:`worlds`'
        ranks): a contact a thread."""
        return self.formed == "stage" and (self.colored or self.grouped)

    @property
    def ABI(self) -> tuple:
        units = self.colored and not self.regions
        if self.world_blocks or self.spread:
            return COLORED_GROUP_ABI if units else GROUP_ABI
        return COLORED_ABI if units else MF_GS_ABI

    @property
    def NATIVE(self) -> str:
        return "sweep_mf_colored_native" if self.colored else "sweep_mf_gs_native"

    @property
    def label(self) -> str:
        return f"the {'colored' if self.colored else 'warp'} matrix-free sweep"

    @property
    def name(self) -> str:
        M, MF = self.max_constraints, self.mf_max_constraints
        return (
            f"sweep_mf_{'colored' if self.colored else 'gs'}_{M}_{MF}_{self.max_world_dofs}"
            f"_s{self.max_row_dofs}"
            + self.places.suffix
            + ("_torsion" if self.torsion else "")
            + ("_weighted" if self.weighted else "")
            + ("_material" if self.material else "")
            + ("_seeded" if self.seeded else "")
            + ("_wide" if self.wide_rows else "")
            + ("_classes" if self.support_classes else "")
            + ("_active" if self.active_rows else "")
            + ("_units" if self.contact_units else "")
            + ("_levels" if self.coordinate_levels else "")
            + (f"_l{self.lanes_per_world}" if self.lanes_per_world != 32 else "")
            + (f"_w{self.block_warps}" if self.block_warps > 1 else "")
            + ("" if self.free_rows else "_line")
            + ("_regions" if self.regions else "")
            + {"inline": "", "ahead": "_formed", "stage": "_records"}[self.formed]
            + ("_tree" if self.one_tree else "")
            + ("_grouped" if self.grouped else "")
            + ("_split" if self.split else "")
            + ("_group" if self.world_blocks else "")
            + "_walk"
            + self.solve_policy.suffix
        )

    source = mf_gs_source

    def bound_bytes(self, staging: Staging) -> int:
        """The stage bytes its launch bound is shaped at: its mapping's (:attr:`line_stage`), else its own."""
        return self.line_stage or staging.bytes

    def staging(self) -> Staging:
        """The block's working set at its places (:func:`mf_gs_staging`): each staged operand with its alternative."""
        return mf_gs_staging(
            self.max_constraints,
            self.mf_max_constraints,
            self.max_world_dofs,
            colored=self.colored,
            grouped=self.grouped,
            regions=self.regions,
            flow=self.regions and self.colored and not self.world_blocks,
            places=self.places,
            traces=self.solve_policy.traces,
            wide_rows=self.wide_rows,
            free_rows=self.free_rows,
            active_rows=self.active_rows,
            coordinate_levels=self.coordinate_levels,
            groups=self.groups,
            lanes=self.lanes,
            split=self.split,
        )

    def dispatch(self, staging: Staging) -> Dispatch:
        # The warp mapping walks the general sweep's cost classes (sweep_levels.COST_CLASSES), the colored one its bucket.
        classes = not (self.colored or self.grouped)
        if self.world_blocks or self.spread:
            return worlds(lanes=self.lanes, ranks="world_blocks")
        return (
            worlds(self.groups, self.lanes, classes) if self.groups > 1 else worlds(lanes=self.lanes, classes=classes)
        )

    def operands(self) -> tuple[RowArgument, ...]:
        return (
            # Where spin rows may appear, or the colored units read them, the friction contacts' records.
            *(
                (RowArgument("friction_metric", wp.array3d[float], after="omega"),)
                if self.torsion or self.colored
                else ()
            ),
            # Where the rows are stored by their support classes, the classes' tables and each row's classes.
            *(
                (
                    RowArgument("row_classes", wp.array2d[int], after="omega"),
                    RowArgument("class_dof", wp.array2d[int], after="omega"),
                    RowArgument("class_count", wp.array[int], after="omega"),
                )
                if self.support_classes
                else ()
            ),
            # Where the units on one coordinate step by level, their schedule.
            *(
                (
                    RowArgument("sparse_contact_level_count", wp.array[int], after="omega"),
                    RowArgument("sparse_contact_level_offsets", wp.array2d[int], after="omega"),
                    RowArgument("sparse_contact_level_units", wp.array2d[int], after="omega"),
                )
                if self.coordinate_levels
                else ()
            ),
            # Where the free bodies' contacts are regions, their stream.
            *((RowArgument(name, dtype, after="omega") for name, dtype in REGION_OPERANDS) if self.regions else ()),
            # A grouped world's tables (row_slots.RowGroups).
            *(
                (
                    RowArgument("row_group_slot", wp.array[int], after="omega"),
                    RowArgument("row_group_colors", wp.array[int], after="omega"),
                    RowArgument("row_group_offsets", wp.array2d[int], after="omega"),
                    RowArgument("row_group_rows", wp.array2d[int], after="omega"),
                )
                if self.grouped
                else ()
            ),
            # Where the world's velocity change lives in global memory, its line a world (RowPlaces.velocity).
            *(
                (RowArgument("sweep_velocity", wp.array2d[float], after="omega"),)
                if self.places.velocity is not Place.SHARED
                else ()
            ),
            # Where a world takes a group of blocks, its job words (WIDE_JOB a world, region_phase).
            *((RowArgument("wide_jobs", wp.array[int], after="omega"),) if self.world_blocks else ()),
            # Where the free rows keep two heads, the one this launch stages (RowPlaces.wide_slots).
            *((RowArgument(FREE_HEAD, int, after="omega"),) if self.places.wide_slots else ()),
            # The warp mapping's worlds' costs, which order its next call (sweep_levels.claim_worlds).
            *(
                (RowArgument("world_cost", wp.array[wp.int32], after="omega"),)
                if not (self.colored or self.grouped)
                else ()
            ),
            *self.law_operands(),
        )

    def work(self, items: int = 0, shared: int | Staging = 0) -> Work:
        """The colored mapping: a block a world of :data:`COLORED_THREADS`, every register a thread may hold (one block
        a multiprocessor). The warp mapping: ``lanes_per_world`` lanes a world on a persistent grid, at the register
        line :data:`WARP_LINE` in blocks of :data:`BLOCK_WARPS` warps."""
        if self.colored or self.grouped:
            return Work(
                int(items), lanes=COLORED_THREADS, block=COLORED_THREADS, registers=REGISTERS_PER_THREAD, shared=shared
            )
        block, lanes = 32 * self.block_warps, self.lanes_per_world
        line = WARP_LINE if self.block_warps > 1 else None
        return Work(int(items), lanes=lanes, block=block, registers=line, persistent=True, shared=shared)

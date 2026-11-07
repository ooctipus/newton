# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""FeatherPGS's general (matrix-free) sweep as a configuration of the sweep emitter (:mod:`.sweep`).

:class:`GeneralSweep` is its configuration, which the one factory builds (:func:`.sweep.sweep_kernel`); its working
set fits the static budget of the device it is built for (:func:`.memory.arch_budget`). The module also holds what
every claim's sweep shares: the world queue's dispatch (:func:`worlds`), the free bodies' rows and walk.
"""

from __future__ import annotations

import dataclasses

import warp as wp

from ...core.execution import Work, device_lanes
from ...core.memory import REGISTERS_PER_THREAD, Buffer, Place, Staging
from . import kernel_template, sweep_units
from .factor.walk import FREE_BODY_WALK, WALK_COORDINATE, WALK_LINK, WalkTrees, walk_down_prelude
from .kernel_template import Dispatch
from .packed_rows import CLASS_SIDES, PACKED_SIDES
from .response_operator import Lanes, SparseFactor, Velocity
from .row_inputs import RowArgument
from .row_laws import ROW_CONTACT, ROW_FRICTION, ROW_JOINT_VELOCITY_LIMIT, ROW_SPEED
from .row_metadata import record_layout, validate_record_coordinates
from .solve_policy import QUIET_LITERAL, block_argmax, sweep_exit
from .sweep import (
    EXACT,
    ITERATION,
    SEQUENTIAL,
    Diagonal,
    Levels,
    RowContext,
    RowLaws,
    SweepConfig,
    UnitZ,
    WorldRows,
    cached_pair_law,
    cached_triple_law,
    contact_unit,
    end_sweep,
    friction_metric,
    friction_metric_law,
    friction_metric_stage,
    impulse_seed,
    indent,
    level_loop,
    rows_block,
    speed_rows_pass,
    velocity_limit_pass,
)

# The row loops leave the velocity-limit and speed rows to the passes at the end of every sweep (the last word).
_DENSE_ADMISSION = f"if (row_type == {ROW_JOINT_VELOCITY_LIMIT}) continue;"
_LINE_ADMISSION = f"if (row_type == {ROW_JOINT_VELOCITY_LIMIT} || row_type == {ROW_SPEED}) continue;"
_MF_ADMISSION = f"if (mf_rt == {ROW_SPEED}) continue;"


def worlds() -> kernel_template.Dispatch:
    """The worlds a sweep launch covers: its claim's bucket of the claims' worlds (:func:`.sweep_levels.claim_worlds`),
    a device list (``general_worlds``, its count first) walked with the grid's stride."""
    return kernel_template.items("world", queue=("general_world_count", "general_world_grid_stride", "general_worlds"))


MF_GS_ABI = kernel_template.params(
    ("world", int),
    ("world_constraint_count", wp.array[int]),
    ("dense_first", wp.array[int]),
    ("dense_phase_bounds", wp.array2d[int]),
    ("speed_first", wp.array[int]),
    ("world_dof_indices", wp.array2d[int]),
    ("rhs_bias", wp.array2d[float]),
    ("row_incident", wp.array2d[float]),
    ("world_diag", wp.array2d[float]),
    ("world_row_w", wp.array2d[float]),
    ("world_impulses", wp.array2d[float]),
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
    ("v_out", wp.array[float]),
)
"""The matrix-free sweep's native ABI: one world per call (the world queue's dispatch). The free bodies' run's rows are
the row store's (the ``world_*`` records) and the packed rows' (``packed_Z``), from the dense capacity of the world's
line; the streamed rows' records the line's (``row_record``, :func:`.row_metadata.pack_row_record`). ``speed_first``:
where the line's speed rows start."""


@dataclasses.dataclass(frozen=True)
class RowPlaces:
    """Where the matrix-free sweep's row operands live, chosen once in the plan
    (:func:`.execution_plan._select_mfgs_storage`): each in the block's shared stage or global memory, with identical
    values.

    Args:
        metadata: The dense rows' metadata: staged, or their records streamed (:class:`.sweep.WorldRows`).
        impulses: The free bodies' impulses.
        dense_impulses: The dense rows' impulses (the line's): staged, or in place.
    """

    metadata: Place = Place.SHARED
    impulses: Place = Place.SHARED
    dense_impulses: Place = Place.SHARED

    def __post_init__(self):
        if {self.metadata, self.impulses, self.dense_impulses} - {Place.SHARED, Place.GLOBAL}:
            raise ValueError(f"row operands live in shared or global memory: {self}")

    def __repr__(self) -> str:
        dense = f", {self.dense_impulses.name}" if self.dense_impulses is not Place.SHARED else ""
        return f"RowPlaces({self.metadata.name}, {self.impulses.name}{dense})"

    @property
    def staging(self) -> str:
        """The dense rows' staging policy (:class:`.sweep.WorldRows`): staged, or their records streamed."""
        return "shared" if self.metadata is Place.SHARED else "record"

    @property
    def suffix(self) -> str:
        """The places' part of a kernel's name."""
        return (
            ("" if self.metadata is Place.SHARED else "_gmeta")
            + ("" if self.impulses is Place.SHARED else "_glambda")
            + ("" if self.dense_impulses is Place.SHARED else "_gline")
        )


STAGED_ROWS = RowPlaces()
"""Every row operand in the block's shared stage, the default."""


def record_first(max_constraints: int, places: RowPlaces) -> int:
    """The line's first row with a record (:func:`.row_metadata.pack_row_record`): the dense rows' first where their
    metadata streams, else the free bodies' (from the dense capacity ``max_constraints``), which always stream."""
    return 0 if places.staging == "record" else int(max_constraints)


ROW_PLACES = (
    STAGED_ROWS,
    RowPlaces(impulses=Place.GLOBAL),
    RowPlaces(Place.GLOBAL),
    RowPlaces(Place.GLOBAL, Place.GLOBAL),
    RowPlaces(Place.GLOBAL, Place.GLOBAL, Place.GLOBAL),
)
"""Every placement of the sweep's row operands (the census's axis)."""


class FreeBodyCoordinates:
    """The way into and out of a free body's walk coordinates, which a matrix-free sweep's speed rows read (native).

    A sweep runs on the velocity change from the predictor in walk coordinates, ``du`` (``s_v``, from zero): a row's
    residual is its right-hand side, which holds its incident velocity ``J v_hat``, plus ``Z^T du``, and its update
    ``du += Z delta``. The sweep adds its change to the walk's ``du`` and the step's one leave walks every tree down
    (:func:`.factor.walk.walk_leave_kernel`). A free body's speed row reads the body's physical velocity mid-sweep: its
    tree (found from its first global coordinate, :class:`.factor.walk.WalkTrees`' ``first_art``) is walked down and
    back in place (:meth:`walk_leave`, :meth:`walk_enter`): the leave and the enter of its one-body class
    (:data:`.factor.walk.FREE_BODY_WALK`, :func:`.factor.walk.walk_down_prelude`), compiled.
    """

    def _walked(self, first: str, call: str) -> str:
        """``call`` (a walk of :func:`.factor.walk.walk_down_prelude` on ``data`` and ``link``) on the tree whose first
        world coordinate is ``first``."""
        return f"""{{
    const int body_art = walk.first_art.data[world_dof_indices.data[dof_map_base + {first}]];
    const int j0 = walk.joint_start.data[body_art], d0 = walk.dof_start.data[body_art];
    const float* data = walk.data.data + (size_t)d0 * {WALK_COORDINATE};
    const float* link = walk.link.data + (size_t)j0 * {WALK_LINK};
    {call};
}}"""

    def walk_leave(self, first: str, values: str) -> str:
        """The walk down of the free body (a one-body tree, :data:`.factor.walk.FREE_BODY_WALK`) whose first world
        coordinate is ``first``, in place on its six ``values``: ``v = P^T L^-T u``."""
        return self._walked(
            first,
            f"WALK_DOWN(0, data, link, [&](const int c) -> float {{ return {values}[c]; }}, "
            "[](const int) -> float { return 0.0f; }, "
            f"[&](const int c, const float q, const float) {{ {values}[c] = q; }}, [](const int, const float*) {{}})",
        )

    def walk_enter(self, first: str, velocity: str, out: str) -> str:
        """The walk in of that free body: ``out = L^T P velocity``."""
        return self._walked(first, f"WALK_ENTER(0, data, link, {velocity}, {out})")


FREE_BODY_COORDINATES = FreeBodyCoordinates()
"""The free bodies' way into and out of walk coordinates in a matrix-free sweep (:class:`FreeBodyCoordinates`)."""

FREE_BODY_WALKS = walk_down_prelude((FREE_BODY_WALK,), ahead=0, enter=True)
"""The leave and the enter of :data:`FREE_BODY_COORDINATES`, compiled into a sweep that visits speed rows, without
prefetch: a one-body walk is no chain whose loads a lane could hold ahead."""


def packed_operator(
    support: int, lanes: Lanes, *, wide: bool = False, rows: WorldRows | None = None, classes: bool = False
) -> SparseFactor:
    """The dense rows of a matrix-free sweep: packed whitened rows of ``support`` entries (:mod:`.packed_rows`), their
    world coordinates from their sides (``PACKED_COORDINATE``; where ``rows`` stream their records, from the sides
    their records hold, ``RECORD_COORDINATE``; with ``classes``, by their support classes too, ``CLASS_*``), or with
    ``wide`` rows every coordinate of the world, and values from the world's first row (``packed_base``), over the
    factor velocity change the sweep stages (``s_v``)."""

    def coordinate(row, entry):
        if wide:
            return f"(world_dof_indices.data[dof_map_base + ({entry})] >= 0 ? ({entry}) : -1)"
        if rows is not None and rows.record:
            prefix = "CLASS_RECORD" if classes else "RECORD"
            return f"{prefix}_COORDINATE({rows.word(row, 0)}, {rows.word(row, 4)}, {entry})"
        if classes:
            return (
                f"CLASS_COORDINATE(row_sides.data[off_dense + ({row})], row_classes.data[off_dense + ({row})], {entry})"
            )
        return f"PACKED_COORDINATE(row_sides.data[off_dense + ({row})], {entry})"

    def value(row, entry):
        return f"packed_Z.data[packed_base + ({row}) * {int(support)} + ({entry})]"

    return SparseFactor(int(support), coordinate, value, lanes, velocity=Velocity("s_v"))


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


COLORED_THREADS = 256
"""Threads of a colored world's block: a color's units beyond them run in turns."""

COLORS = Levels("color_count.data[world]", lambda color: f"color_offset.data[off_color + {color}]", chunk=256)
"""A colored world's free-body units by color (the exact level law, :mod:`.sweep_units`), a thread a unit of the block:
256 colors' offsets staged at a time, since a world's colors are bounded by its units, not by the block."""

COLORED_ABI = kernel_template.params(
    *MF_GS_ABI[:-1],
    ("world_slot", wp.array[int]),
    ("color_count", wp.array[int]),
    ("color_offset", wp.array2d[int]),
    ("color_slot", wp.array3d[int]),
    ("slot_data", wp.array3d[float]),
    ("slot_words", wp.array3d[int]),
    ("slot_metric", wp.array3d[float]),
    ("v_out", wp.array[float]),
)
"""The colored mapping's native ABI: the matrix-free sweep's, then each world's storage slot, the worlds' color counts, their slots' colors and staged units (:data:`.sweep_units.WORLD_FIELDS`, :data:`.sweep_units.UNIT_FIELDS`,
:data:`.sweep_units.SLOT_FIELDS`) before ``v_out``."""


_UNIT_ROWS = (("n", "unit_jv0", "unit_dn"), ("t1", "unit_jv1", "unit_dt1"), ("t2", "unit_jv2", "unit_dt2"))
"""A colored unit's rows: each one's partial sum, velocity term and deferred change, at ``E``'s ``12 r``."""
_SPIN_ROW = ("spin", "unit_jv3", "unit_ds")
"""Under contact torsion the unit's fourth row, its spin row."""


def _unit_rows(torsion: bool) -> tuple:
    return (*_UNIT_ROWS, _SPIN_ROW) if torsion else _UNIT_ROWS


_PAIR_METRIC = ("metric.v0", "metric.v1", "metric.v2[0]", "metric.v2[1]")
"""A colored friction unit's cached pair metric (``ProjectionLaw::friction_pair_metric``): the two eigenvalues, the
larger's axis."""
TRIPLE_METRIC = 8
"""Under contact torsion a friction unit's cached metric takes eight floats: the friction triple's
(``ProjectionLaw::friction_ball_metric``: three eigenvalues and an unused value, the eigenbasis as a rotation
quaternion), or a unit without a spin row the pair's (:data:`_PAIR_METRIC`) and four unused."""
_PAIR_COUPLINGS = (("gram_n1", 0, 1), ("gram_n2", 0, 2))
"""The Gram couplings a colored friction unit caches after its pair's metric: each one's name and rows (``E``'s
``12 r``), the pair's tangents with the normal."""
_SPIN_COUPLINGS = (("gram_n3", 0, 3),)
"""Under contact torsion the spin row's coupling with the normal, cached after the pair's."""
_TRIPLE_COUPLINGS = (("unit_cross", 1, 2), ("unit_cross13", 1, 3), ("unit_cross23", 2, 3))
"""Under contact torsion the friction rows' couplings, which form the unit's friction metric before the sweeps."""


def _unit_couplings(torsion: bool) -> tuple[int, tuple]:
    """A colored unit's cached Gram couplings and their start in its metric: after the friction pair's metric
    (:data:`_PAIR_METRIC`), or under contact ``torsion`` after the friction metric's :data:`TRIPLE_METRIC` floats, with
    the spin row's."""
    return (TRIPLE_METRIC, (*_PAIR_COUPLINGS, *_SPIN_COUPLINGS)) if torsion else (len(_PAIR_METRIC), _PAIR_COUPLINGS)


def _unit_terms(pad: str, *, torsion: bool = False) -> str:
    """Native statements, after a unit's ``E``, ``metric``, ``rows`` and ``dof_a``, ``dof_b`` are declared: the unit's
    velocity at its staged rows' entries (``velocities``, :func:`_staged_rows`), its rows' velocity terms from it (each
    side's six products, then the sides added) and the cached Gram couplings of the pair with the normal; under
    ``torsion`` the spin row's term too and its coupling with the normal (:attr:`.sweep_units.SlotLayout.metric`)."""
    rows = _unit_rows(torsion)
    start, couplings = _unit_couplings(torsion)
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
        f"const float {', '.join(f'{name} = metric[{start + f}]' for f, (name, _, _) in enumerate(couplings))};",
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


def _colored_units(free: WorldRows, laws: RowLaws, M_MF: int, units: int) -> str:
    """A color's units, one thread each, at block stride: each loads its slot's fields at once
    (:data:`.sweep_units.SLOT_FIELDS`, field by field across the color's ``units`` slots) and runs the free-body row
    law, unrolled over the unit's rows, on the world's staged velocity. No other unit of the color shares a body."""
    unit_rows = WorldRows(M_MF, staging="record", free=True, stream=free.stream, lanes=Lanes(1))
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
        # row (_unit_terms), the normal's Gram couplings and the friction metric (the pair's, or the triple's with a
        # spin row) from the unit's cache formed once per launch (_colored_metrics), one pass over the unit's
        # coordinates after its rows (_unit_apply).
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
        pair_law="FRICTION_PAIR_CACHED(place, ",
        triple_law="FRICTION_TRIPLE_CACHED(place, ",
    )
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
            "            // A friction pair's law from its unit's cached metric (formed once per launch, before the sweeps), which",
            "            // the unit loads with its fields"
            + ("; under contact torsion the triple's law likewise, with a spin row." if torsion else "."),
            f"            float metric[{layout.metric}];",
            cached_pair_law("FRICTION_PAIR_CACHED", "metric[0]", "metric[1]", ("metric[2]", "metric[3]"), " " * 12),
            *(
                (cached_triple_law("FRICTION_TRIPLE_CACHED", " " * 12, metric=tuple(f"metric[{f}]" for f in range(8))),)
                if torsion
                else ()
            ),
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
    visit = f"""                    const int4 slot = __ldg(reinterpret_cast<const int4*>(&color_slot.data[(off_unit + place) * 4]));
                    const int unit_row = slot.x, rows = slot.y, dof_a = slot.z, dof_b = slot.w;
                    const int slot_base = unit_slot * {fields * units} + place;
                    const int word_base = unit_slot * {words * units} + place;
                    // Every field of the slot and its metric at once, beside the slot's own load: one round of
                    // memory latency per unit. A unit without a friction pair never reads its pair's fields.
                    float E[{fields}];
                    int W[{words}];
#pragma unroll
                    for (int f = 0; f < {fields}; ++f) E[f] = slot_data.data[slot_base + f * {units}];
#pragma unroll
                    for (int f = 0; f < {words}; ++f) W[f] = slot_words.data[word_base + f * {units}];
#pragma unroll
                    for (int f = 0; f < {layout.metric}; ++f)
                        metric[f] = slot_metric.data[unit_slot * {layout.metric * units} + place + f * {units}];
{_unit_terms(" " * 20, torsion=torsion)}
{unit}
{_unit_apply(" " * 20, torsion=torsion)}"""
    return f"{ring_sum}\n{COLORS.wrap(visit, ' ' * 12)}"


def _colored_diagonal(r: int) -> str:
    """A colored unit's row ``r``'s diagonal ``A_rr`` from its stored inverse (``W[3 r]``; zero for a zero one)."""
    return f"(__int_as_float(W[{3 * r}]) > 0.0f ? 1.0f / __int_as_float(W[{3 * r}]) : 0.0f)"


def _colored_pair_metric(pad: str) -> str:
    """Native statements, after a unit's ``W`` and its pair's coupling ``unit_cross`` are declared: its friction pair's
    metric (``ProjectionLaw::friction_pair_metric`` of the first tangent's diagonal, the pair's coupling and the
    second tangent's diagonal), formed exactly as the visit's pair law forms it."""
    return f"{pad}const auto metric = ProjectionLaw::friction_pair_metric({_colored_diagonal(1)}, unit_cross, {_colored_diagonal(2)});"


def _colored_friction_metric(layout: sweep_units.SlotLayout, pad: str) -> list[str]:
    """Native statements under contact torsion, after a unit's ``E``, ``W``, ``slot`` and its rows' couplings are
    declared: its friction metric (``friction_metric``, :data:`TRIPLE_METRIC` floats), the friction triple's
    (``ProjectionLaw::friction_ball_metric`` of its friction rows' block) with a spin row, else the pair's
    (``ProjectionLaw::friction_pair_metric``), each formed from the diagonals the visit reads."""
    lines = [
        f"float friction_metric[{TRIPLE_METRIC}] = {{}};",
        "if ((slot.y & 7) == 4) {",
        "    const auto triple = ProjectionLaw::friction_ball_metric(",
        f"        wp::mat33({_colored_diagonal(1)}, unit_cross, unit_cross13,",
        f"                  unit_cross, {_colored_diagonal(2)}, unit_cross23,",
        f"                  unit_cross13, unit_cross23, {_colored_diagonal(3)}),",
        f"        E[{layout.mu}], E[{layout.spin_mu}]);",
        "    const wp::quat basis = wp::quat_from_matrix(triple.v1);",
        "    for (int f = 0; f < 3; ++f) friction_metric[f] = triple.v0[f];",
        "    for (int f = 0; f < 4; ++f) friction_metric[4 + f] = basis[f];",
        "} else {",
        _colored_pair_metric("    "),
        *(f"    friction_metric[{f}] = {value};" for f, value in enumerate(_PAIR_METRIC)),
        "}",
    ]
    return [pad + line for line in lines]


def _colored_metrics(units: int, *, torsion: bool = False) -> str:
    """Before the sweeps, every thread forms the cached metric of the world's staged friction units at block stride
    and stores it in its slot's metric cache (``slot_metric``, field by field across the units), which the unit's law
    and terms read at every visit: a friction pair's metric (the two eigenvalues and the larger's axis), or
    under contact ``torsion`` the friction metric of :func:`_colored_friction_metric`, then its Gram couplings
    (:func:`_unit_couplings`)."""
    layout = sweep_units.slot_layout(torsion)
    fields, words = layout.fields, layout.words
    couplings = _unit_couplings(torsion)[1]
    names = [name for name, _, _ in couplings]
    head = [
        "    {",
        "        const int staged = color_offset.data[off_color + color_count.data[world]];",
        "        for (int p = tid; p < staged; p += blockDim.x) {",
        "            const int4 slot = __ldg(reinterpret_cast<const int4*>(&color_slot.data[(off_unit + p) * 4]));",
        # A friction unit: three rows, or four with its spin row where spin rows may appear (sweep.spin_row).
        "            if ((slot.y & 7) < 3) continue;"
        if torsion
        else f"            if ((slot.y & 3) != {layout.rows}) continue;",
        "            const int dof_a = slot.z, dof_b = slot.w;",
        f"            const int slot_base = unit_slot * {fields * units} + p;",
    ]
    stage = [
        "#pragma unroll",
        f"            for (int f = 0; f < {fields}; ++f) E[f] = slot_data.data[slot_base + f * {units}];",
    ]
    # The friction rows' couplings too, which form the friction metric with the rows' diagonals: the pair's, and under
    # torsion the spin row's with each tangent.
    couplings = (*couplings, *(_TRIPLE_COUPLINGS if torsion else _TRIPLE_COUPLINGS[:1]))
    metric = _colored_friction_metric(layout, " " * 12) if torsion else [_colored_pair_metric(" " * 12)]
    values = [*((f"friction_metric[{f}]" for f in range(TRIPLE_METRIC)) if torsion else _PAIR_METRIC), *names]
    rows = [
        *head,
        f"            const int word_base = unit_slot * {words * units} + p;",
        f"            float E[{fields}];",
        f"            int W[{words}];",
        *stage,
        "#pragma unroll",
        f"            for (int f = 0; f < {layout.ring}; ++f) W[f] = slot_words.data[word_base + f * {units}];",
        "            // The Gram couplings of the unit's rows: per coordinate, body a's then b's products.",
    ]

    def products(side: int) -> list[str]:
        entry = lambda row: f"E[{12 * row + 6 * side} + k]" if 12 * row + 6 * side else "E[k]"  # noqa: E731
        return [f"                    {name} += {entry(a)} * {entry(b)};" for name, a, b in couplings]

    rows += [
        f"            float {', '.join(f'{name} = 0.0f' for name, _, _ in couplings)};",
        "            for (int k = 0; k < 6; ++k) {",
        "                if (dof_a >= 0) {",
        *products(0),
        "                }",
        "                if (dof_b >= 0) {",
        *products(1),
        "                }",
        "            }",
        *metric,
        f"            const int at = unit_slot * {layout.metric * units} + p;",
        *(
            f"            slot_metric.data[at{f' + {f * units}' if f else ''}] = {value};"
            for f, value in enumerate(values)
        ),
        "        }",
        "    }",
        "    __syncthreads();",
    ]
    return "\n".join(rows)


def mf_gs_staging(
    max_constraints: int,
    mf_max_constraints: int,
    max_world_dofs: int,
    *,
    colored: bool = False,
    places: RowPlaces = STAGED_ROWS,
    traces: bool = False,
    wide_rows: bool = False,
    free_rows: bool = True,
) -> Staging:
    """The shared working set of a world's matrix-free sweep block (:func:`mf_gs_source` declares it): the world's
    velocity, its dense rows' impulses and staged metadata, the free bodies' run's impulses (``free_rows``) unless they
    stream, and ``colored`` the colors' staged offsets (and its trace's warp maxima)."""
    dense = WorldRows(max_constraints, staging=places.staging, stream=places.dense_impulses is Place.GLOBAL)
    free = WorldRows(
        mf_max_constraints, staging="record", free=True, stream=places.impulses is Place.GLOBAL or not free_rows
    )
    levels = ()
    if colored:
        levels = COLORS.buffers()
        if traces:
            levels += block_argmax(COLORED_THREADS // 32, warp="warp", lane="lane")[1]
    return Staging(
        (
            Buffer("s_v", "float", max_world_dofs),
            *((Buffer("s_sweep_v", "float", max_world_dofs),) if wide_rows else ()),
            *dense.buffers(),
            *free.buffers(),
            *levels,
        ),
        threads=COLORED_THREADS if colored else 32,
    )


def mf_gs_source(sweep: GeneralSweep) -> tuple[str, Staging]:
    """The native body of the matrix-free Gauss-Seidel sweep: a world's dense rows (:func:`packed_operator`, packed
    whitened rows of ``max_row_dofs`` entries, by default as many as the world's coordinates) and its free-body rows
    (:func:`free_body_rows`), then its end-of-sweep passes, on the world's velocity change from the predictor in walk
    coordinates (:class:`FreeBodyCoordinates`), from zero, or with ``seeded`` from the response of the impulses the
    rows carry (:func:`.sweep.impulse_seed`). The launch adds it to the walk's ``du`` (:attr:`.factor.walk.WalkTrees.du`),
    which the step's one leave walks down onto ``v_out`` (:func:`.factor.walk.walk_leave_kernel`); a free body's speed
    row reads the body's velocity through its walk (:class:`FreeBodyCoordinates`).

    Every switch is the configuration's (:class:`GeneralSweep`).

    Returns the body and its staging.
    """
    places, solve_policy = sweep.places, sweep.solve_policy
    contact_torsion, seeded, wide_rows = sweep.torsion, sweep.seeded, sweep.wide_rows
    free_rows, speed_rows, max_row_dofs = sweep.free_rows, sweep.speed_rows, sweep.max_row_dofs
    colored = sweep.colored
    lanes = sweep.lanes
    M_D, M_MF, D = sweep.max_constraints, sweep.mf_max_constraints, sweep.max_world_dofs
    # The row store's line: the dense rows, then the free bodies' from M_D (:mod:`.row_records`).
    P = int(M_D + M_MF if sweep.pitch is None else sweep.pitch)
    row_lanes = Lanes(lanes)
    laws = sweep.laws
    dense = WorldRows(M_D, staging=places.staging, stream=places.dense_impulses is Place.GLOBAL, lanes=row_lanes)
    free = WorldRows(M_MF, staging="record", free=True, stream=places.impulses is Place.GLOBAL, lanes=row_lanes)
    dense_op = packed_operator(max_row_dofs, row_lanes, wide=wide_rows, rows=dense, classes=sweep.support_classes)
    free_op = free_body_rows(row_lanes, max_row_dofs)
    staging = mf_gs_staging(
        M_D,
        M_MF,
        D,
        colored=colored,
        places=places,
        traces=solve_policy.traces,
        wide_rows=wide_rows,
        free_rows=free_rows,
    )

    dense_count = "int m_dense = world_constraint_count.data[world];"
    dense_loads = indent(dense.loads("i"), "        ")
    mf_load = free.loads("i")
    mf_store = free.stores("i")

    passes = []
    if sweep.has_dense_velocity_limit_rows:
        passes.append(velocity_limit_pass(dense, dense_op, laws, pad="        "))
    # The free bodies' speed rows leave and enter by the walk, as their rows were written (walk_rows): the run's, or
    # the line's last rows, each naming its body by its side.
    if free_rows:
        passes.append(speed_rows_pass(free, free_op, laws, walk=FREE_BODY_COORDINATES, pad="        "))
    if speed_rows:
        body = (
            (
                "const wp::vec2i speed_sides = row_sides.data[off_dense + i];",
                "const int speed_body = articulation_world_dof_offset.data[speed_sides[0]];",
            ),
            "speed_body",
            dense_op.element("i", "PACKED_FIRST(speed_sides, speed_sides[0]) + 3"),
        )
        passes.append(
            speed_rows_pass(
                dense,
                dense_op,
                laws,
                walk=FREE_BODY_COORDINATES,
                start="speed_start",
                end="m_dense",
                body=body,
                pad="        ",
            )
        )

    # Where spin rows may appear, every friction triple's metric is formed once per launch, at its first loaded visit
    # (sweep.friction_metric_stage), and its later visits only project: the dense rows' in either mapping, the free
    # bodies' in the warp's (the colored units cache theirs in their slots, _colored_metrics).
    dense_metric = free_metric = None
    friction_laws = metric_stage = ""
    if contact_torsion:
        dense_metric, dense_record = "DENSE_TRIPLE", friction_metric

        def free_record(row: str) -> str:
            # The free bodies' rows follow the dense ones in the line.
            return friction_metric(row, str(M_D))

        friction_laws = friction_metric_law(dense_metric, dense_record, "    ") + "\n"
        metric_stage = (
            friction_metric_stage(
                *dense.scan(), lambda row: dense.spin_of(row, "m_dense"), end="m_dense", record=dense_record, pad="    "
            )
            + "\n"
        )
        if free_rows and not colored:
            free_metric = "FREE_TRIPLE"
            friction_laws += friction_metric_law(free_metric, free_record, "    ") + "\n"
            stage = friction_metric_stage(
                *free.scan(),
                lambda row: free.spin_of(row, "mf_contact_end"),
                end="mf_contact_end",
                record=free_record,
                pad="    ",
            )
            metric_stage += stage + "\n"

    if colored:
        # The block's vote: every warp's rows and units (a barrier); the first thread records the solve.
        quiet = "!__syncthreads_or(iteration_changed != 0)"
        sweep_end = sweep_exit(
            solve_policy,
            quiet=quiet,
            iteration=ITERATION,
            lane="tid",
            argmax="BLOCK_ARGMAX",
        )
        head = "    const int lane = LANE;\n    const int tid = (int)threadIdx.x;\n    const int warp = tid >> 5;"
        first_warp, end_first_warp, barrier = "    if (warp == 0) {\n", "    }\n", "__syncthreads();"
        units = sweep_units.unit_capacity(M_MF)
        free_phase = _colored_units(free, laws, M_MF, units)
        trace_reduction = ""
        if solve_policy.traces:
            trace_reduction = block_argmax(COLORED_THREADS // 32, warp="warp", lane="lane")[0]
        # The world's storage slot (a colored world has one): its colors and staged units.
        offsets = (
            "\n    const int unit_slot = world_slot.data[world];"
            f"\n    const int off_unit = unit_slot * {units};"
            f"\n    const int off_color = unit_slot * {units + 1};"
        )
    else:
        sweep_end = end_sweep(solve_policy, iteration=ITERATION)
        head = "    const int lane = LANE;"
        first_warp = end_first_warp = ""
        barrier = "SYNC();"
        free_phase = ""
        if free_rows:
            free_phase = rows_block(free, free_op, laws, admission=_MF_ADMISSION, metric=free_metric, pad="    ")
        offsets = trace_reduction = ""

    thread, threads = ("tid", "(int)blockDim.x") if colored else ("lane", "STRIDE")
    metric_build = _colored_metrics(sweep_units.unit_capacity(M_MF), torsion=contact_torsion) if colored else ""
    seed = ""
    if seeded:
        seeds = (
            impulse_seed(
                "m_dense",
                dense.peek("i"),
                f"({dense.kind('i', scan=True)})",
                dense_op.axpy((("row", "seed"),), "        ", row="i"),
                row_lanes.sync(),
                declare=tuple(line for line in dense.visit("i").split("\n") if line),
            ),
        )
        seeds += (
            (
                impulse_seed(
                    "m_mf",
                    free.load("i"),
                    free.kind("i"),
                    free_op.axpy((("row", "seed"),), "        ", row="i"),
                    row_lanes.sync(),
                    skip=ROW_SPEED,
                    declare=(
                        *free.visit("i").split("\n"),
                        "const int packed_dofs = rec_a.x;",
                        "const int dof_a = packed_dofs >> 16, dof_b = (packed_dofs << 16) >> 16;",
                    ),
                ),
            )
            if free_rows
            else ()
        )
        seed = f"{first_warp}{chr(10).join(seeds)}\n{end_first_warp}    {barrier}\n"
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
    admission = _LINE_ADMISSION if speed_rows else _DENSE_ADMISSION
    dense_phase = rows_block(dense, dense_op, laws, admission=admission, metric=dense_metric, pad="        ")
    sweeps = level_loop(
        (
            f"{first_warp}{dense_phase}\n{end_first_warp}" + ("        __syncthreads();" if colored else ""),
            indent(free_phase, "    "),
        ),
        (f"{first_warp}{chr(10).join(passes)}\n{end_first_warp}",),
        exit=sweep_vote + sweep_end,
    )
    # The streamed rows' records (row_record, row_metadata.record_layout).
    free_records, line_records = record_layout(record_first(M_D, places), M_D, P)
    records = f"    const int off_rec_mf = line * {line_records} + {free_records};"
    if dense.record:
        records = f"    const int off_rec_dense = line * {line_records};\n" + records
    # The way back after the sweeps: the change in the walk's coordinates to the step's one leave (WalkTrees.du,
    # factor.walk.walk_leave_kernel).
    # The free bodies' run's rows (free_rows): their count, impulses' staging and publication.
    mf_count, free_view, free_loads, free_stores = "int m_mf = 0;", "", "", ""
    if free_rows:
        mf_count = "int m_mf = mf_constraint_count.data[world];"
        free_view = free.declaration()
        free_loads = f"""
    for (int i = lane; i < m_mf; i += STRIDE) {{
        {mf_load}
    }}"""
        free_stores = f"""
    for (int i = lane; i < m_mf; i += STRIDE) {{
        {mf_store}
    }}"""
    if speed_rows:
        dense_count += "\n    const int speed_start = speed_first.data[world];"
    dense_view = dense.declaration() + "\n" if dense.stream else ""
    body = f"""{head}

    {dense_count}
    {mf_count}
    if (m_dense == 0 && m_mf == 0) return;
    if (m_dense > {M_D}) m_dense = {M_D};
    if (m_mf > {M_MF}) m_mf = {M_MF};
    int mf_contact_end = mf_contact_rows_end.data[world];
    if (mf_contact_end > m_mf) mf_contact_end = m_mf;

    int dof_map_base = world * {D};
    const int line = dense_first.data[world];
    __builtin_assume(line >= 0);
    int off_dense = line * {P};
    int off_mf = off_dense + {M_D};
{records}
    const size_t packed_base = (size_t)line * {P * int(max_row_dofs)};
    const size_t free_base = packed_base + {M_D * int(max_row_dofs)};{offsets}

{staging.declarations("    ")}
{dense_view}{free_view}{trace_reduction}
{friction_laws}
{first_warp}    for (int i = lane; i < m_dense; i += STRIDE) {{
{dense_loads}
    }}{free_loads}
    for (int d = lane; d < {D}; d += STRIDE) s_v[d] = 0.0f;
{end_first_warp}    {barrier}
{seed}{first_warp}{metric_stage}{end_first_warp}{metric_build}

{sweeps}

    {barrier}
{first_warp}    for (int d = lane; d < {D}; d += STRIDE) {{
        int global_dof = world_dof_indices.data[dof_map_base + d];
        if (global_dof >= 0) walk.du.data[global_dof] += s_v[d];
    }}
    for (int i = lane; i < m_dense; i += STRIDE) {{
        {dense.stores("i")}
    }}{free_stores}
{end_first_warp}"""
    sides = (PACKED_SIDES, *((CLASS_SIDES,) if sweep.support_classes else ()))
    return sweep.native(body, lanes=lanes, rows=sides, walks=(FREE_BODY_WALKS,)), staging


@dataclasses.dataclass(frozen=True)
class GeneralSweep(SweepConfig):
    """The general (matrix-free Gauss-Seidel) sweep's configuration: a world's dense rows (at most
    ``max_constraints``, packed whitened rows of ``max_row_dofs`` entries, by default as many as the world's
    ``max_world_dofs`` coordinates) and its free-body rows (at most ``mf_max_constraints``) on the world's velocity
    change in walk coordinates (:func:`mf_gs_source`).

    Its switches, each the plan's from the class's shape, the placement and the laws (:meth:`.sweep_plan.Sweep.build`):

    - the free bodies' level law (``levels``, :class:`.sweep.LevelLaw`): sequential, the world's warp visits its rows
      in row order; exact (colored), a block visits the free bodies' colors in order and a color's units at once (the
      unit builder's rule picks the worlds, :class:`.sweep_units.ColoredRule`);
    - the row record: ``places`` (:class:`RowPlaces`, the plan's storage choice): the dense rows' metadata staged or
      their records streamed; the free rows' records always stream;
    - the support class: ``max_row_dofs`` (the placement's packed support), ``support_classes`` (the rows' sides by
      their classes, :func:`.packed_rows.row_classes`), ``wide_rows`` (the bilateral block's), and the free bodies' 6 +
      6 class: ``free_rows`` their own run, else rows of the line, their ``speed_rows`` the line's last;
    - ``has_dense_velocity_limit_rows``: the velocity limits' last word; ``seeded``: the seed; ``pitch``: the line.

    The warp mapping is a warp a world on CUDA and one thread on the CPU (:meth:`work`); the colored one a block of
    :data:`COLORED_THREADS`. Each walks its bucket of the claims' worlds (:func:`worlds`).
    """

    max_constraints: int
    mf_max_constraints: int
    max_world_dofs: int
    device_arch: str
    _: dataclasses.KW_ONLY
    max_row_dofs: int | None = None
    places: RowPlaces = STAGED_ROWS
    has_dense_velocity_limit_rows: bool = True
    seeded: bool = False
    wide_rows: bool = False
    support_classes: bool = False
    pitch: int | None = None
    free_rows: bool = True
    speed_rows: bool = False

    def __post_init__(self):
        if self.levels.law not in (SEQUENTIAL, EXACT):
            raise ValueError("the general sweep's free bodies step in order or by exact levels (colored)")
        if self.colored and not self.free_rows:
            raise ValueError("the colored mapping sweeps the free bodies' run: it needs free_rows")
        validate_record_coordinates(self.max_world_dofs)
        if self.material and not self.weighted:
            raise ValueError("material contacts step by their weights: their sweeps are weighted")
        dofs = self.max_world_dofs if self.max_row_dofs is None else self.max_row_dofs
        object.__setattr__(self, "max_row_dofs", int(dofs))

    @property
    def colored(self) -> bool:
        return self.levels.law == EXACT

    @property
    def lanes(self) -> int:
        """A world's lanes: the colored mapping's block of warps, the warp mapping's warp on CUDA and one thread on the
        CPU (:func:`.execution.device_lanes`), which runs the world on either device."""
        return 32 if self.colored else device_lanes(self.device_arch)

    @property
    def ABI(self) -> tuple:
        return COLORED_ABI if self.colored else MF_GS_ABI

    @property
    def NATIVE(self) -> str:
        return "sweep_mf_colored_native" if self.colored else "sweep_mf_gs_native"

    @property
    def label(self) -> str:
        return f"the {'colored' if self.colored else 'warp'} matrix-free sweep"

    @property
    def name(self) -> str:
        M, MF, pitch = self.max_constraints, self.mf_max_constraints, self.pitch
        return (
            f"sweep_mf_{'colored' if self.colored else 'gs'}_{M}_{MF}_{self.max_world_dofs}"
            f"_s{self.max_row_dofs}"
            f"_densevlim{int(self.has_dense_velocity_limit_rows)}"
            + self.places.suffix
            + ("_torsion" if self.torsion else "")
            + ("_weighted" if self.weighted else "")
            + ("_material" if self.material else "")
            + ("_seeded" if self.seeded else "")
            + ("_wide" if self.wide_rows else "")
            + ("_classes" if self.support_classes else "")
            + (f"_p{pitch}" if pitch is not None and pitch != M + MF else "")
            + ("" if self.free_rows else "_line")
            + ("_speed" if self.speed_rows else "")
            + "_walk"
            + self.solve_policy.suffix
        )

    source = mf_gs_source

    def dispatch(self, staging: Staging) -> Dispatch:
        return worlds()

    def operands(self) -> tuple[RowArgument, ...]:
        return (
            # Where spin rows may appear, every friction contact's metric (sweep.friction_metric_stage).
            *((RowArgument("friction_metric", wp.array3d[float], after="omega"),) if self.torsion else ()),
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
            *self.law_operands(),
        )

    def work(self, items: int = 0, shared: int = 0) -> Work:
        """The colored mapping: a block a world of :data:`COLORED_THREADS`, every register a thread may hold (one block
        a multiprocessor). The warp mapping: a warp a world on a persistent grid; no register line."""
        if self.colored:
            return Work(int(items), lanes=COLORED_THREADS, registers=REGISTERS_PER_THREAD, shared=int(shared))
        return Work(int(items), lanes=32, block=32, persistent=True)

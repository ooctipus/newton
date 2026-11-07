# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The paired factor sweep (case K5) as a configuration of the sweep emitter (:mod:`.sweep`).

:class:`PairedSweep`, built by the one factory (:func:`.sweep.sweep_kernel`): one warp per world of an articulation and
a free body, rows in factor coordinates (:class:`.response_operator.SparseFactor`, dense, in registers) on the velocity
change from the predictor, which leaves per component, under one of two schedules: row by row, or contact units. Its
working set fits the static budget of the device it is built for (:func:`.memory.arch_budget`).
"""

from __future__ import annotations

import dataclasses
from typing import ClassVar

import warp as wp

from ...core.execution import Work, device_lanes
from ...core.memory import Buffer, Staging
from . import kernel_template
from .device_preludes import RING, SHARED_STORAGE, group_native, lane_group
from .factor.walk import WalkTrees
from .kernel_template import Dispatch
from .law_source import law_preludes
from .projection_laws import CONE_LAW, PROJECTION_LAW
from .response_operator import Lanes, LaneVelocity, SparseFactor, Velocity
from .row_inputs import RowArgument
from .sweep import (
    FRICTION,
    ITERATION,
    TORSION,
    UNIT_HELD,
    Diagonal,
    Pipelined,
    RowContext,
    RowLaws,
    StagedRows,
    SweepConfig,
    UnitStride,
    UnitZ,
    contact_unit,
    end_sweep,
    friction_metric,
    friction_metric_law,
    friction_metric_stage,
    impulse_seed,
    indent,
    level_loop,
    material_arguments,
    pair_radius,
    pair_update,
    row_step,
    row_sums,
    row_update,
    speed_rows_pass,
    unit_held_loads,
    weight_arguments,
)
from .sweep_routes import FREE_BODY_COORDINATES, FREE_BODY_WALKS

# ---------------------------------------------------------------------------------------------------------------------
# The paired factor sweep: rows in factor coordinates, a warp per world, three schedules
# ---------------------------------------------------------------------------------------------------------------------

PAIRED_FACTOR_ABI = kernel_template.params(
    ("world", int),
    ("world_constraint_count", wp.array[int]),
    ("dense_first", wp.array[int]),
    ("dense_phase_bounds", wp.array2d[int]),
    ("world_dof_indices", wp.array2d[int]),
    ("rhs_bias", wp.array2d[float]),
    ("row_incident", wp.array2d[float]),
    ("world_diag", wp.array2d[float]),
    ("world_impulses", wp.array2d[float]),
    ("factor_rows", wp.array3d[float]),
    ("world_row_type", wp.array2d[int]),
    ("world_row_parent", wp.array2d[int]),
    ("world_row_mu", wp.array2d[float]),
    ("speed_first", wp.array[int]),
    ("secondary_offset_by_world", wp.array[int]),
    ("iterations", int),
    ("omega", float),
    ("v_out", wp.array[float]),
)
"""The paired factor sweep's native ABI: one world per warp, two per block (a tail-guarded tiled dispatch); the line's
speed rows from ``speed_first``, its free body's first world coordinate ``secondary_offset_by_world``."""

_PAIRED_WORLDS_PER_BLOCK = 2

PAIRED_REGISTER_LINE = 80
"""The paired factor sweep's register line: 12 blocks of two worlds a multiprocessor. It is no cap (the layer's
``Work.cap``): the sweep is bound only where its shared memory keeps no more blocks than the line, as the tiers are,
which the line's speed rows need: unbounded, kuka's sweep with them compiles at 96 registers (10 blocks of the 11 its
shared memory keeps on sm_120), within the line at 80 with the same 24-byte stack."""


_ROWS = "factor_rows"
"""The world's rows in factor coordinates: row ``i``'s ``D`` entries from ``row_base + i * D``."""


def _registers(lanes: Lanes, **rows: str):
    """The rows' entries a visit acts on, by name, in the registers the schedule loaded them into (``SparseFactor``'s
    ``factor`` with ``registers``): a lane's own entry, or the one lane's row, an entry a register."""
    if lanes.parallel:
        return lambda name, entry: rows[name]
    return lambda name, entry: f"{rows[name]}[{entry}]"


def _rows(dofs: int, lanes: Lanes) -> SparseFactor:
    """The paired rows: dense (entry ``l`` at coordinate ``l``, at most a warp), the visited row in ``row_z``, its
    friction sibling in ``sibling_z`` and its contact's spin row in ``spin_z``, ``u`` in ``factor_velocity``; each
    velocity sum reduces its terms' magnitude beside it. Lane ``l`` holds entry and coordinate ``l`` of each; one lane
    holds them all (:func:`_hold`)."""
    return SparseFactor(
        dofs,
        lambda row, entry: entry,
        _registers(lanes, row="row_z", sib="sibling_z", spin="spin_z"),
        lanes,
        velocity=LaneVelocity("factor_velocity") if lanes.parallel else Velocity("factor_velocity"),
        registers=True,
        magnitudes=True,
    )


def _load(op: SparseFactor, row: str) -> str:
    """This lane's entry of row ``row`` (zero past the rows' coordinates)."""
    lane, D = op.lanes.lane, op.support
    return f"{lane} < {D} ? {_ROWS}.data[row_base + ({row}) * {D} + {lane}] : 0.0f"


def _hold(op: SparseFactor, name: str, row: str, condition: str = "") -> str:
    """Declare the register ``name`` with row ``row``'s entries (zero where ``condition`` fails): this lane's entry, or
    the one lane's row, an entry a register."""
    if op.lanes.parallel:
        value = f"{condition} ? ({_load(op, row)}) : 0.0f" if condition else _load(op, row)
        return f"const float {name} = {value};"
    D = op.support
    value = f"{_ROWS}.data[row_base + ({row}) * {D} + e]"
    value = f"{condition} ? {value} : 0.0f" if condition else value
    return f"float {name}[{D}];\nfor (int e = 0; e < {D}; ++e) {name}[e] = {value};"


def _consume(op: SparseFactor, row: str, end: str, indent: str) -> str:
    """Take row ``row``'s entry from the pipeline register ``pre_z`` (``row_z``) and load row ``row + 1``'s (before
    ``end``); the one lane loads its row ``row`` whole."""
    lane, D = op.lanes.lane, op.support
    if not op.lanes.parallel:
        return "\n".join(indent + line for line in _hold(op, "row_z", row).split("\n"))
    return (
        f"{indent}const float row_z = pre_z;\n"
        f"{indent}if ({row} + 1 < {end} && {lane} < {D})\n"
        f"{indent}    pre_z = {_ROWS}.data[row_base + ({row} + 1) * {D} + {lane}];"
    )


def paired_factor_source(sweep: PairedSweep) -> tuple[str, Staging]:
    """The native body of the paired factor sweep ``sweep`` (:class:`PairedSweep`'s source): a world's rows in the
    factor coordinates of its articulation and free body, a warp per world, the velocity change from the predictor, ``u = L^T dv``, from zero in a register per
    lane, or with ``seeded`` from the response of the impulses the rows carry (:func:`.sweep.impulse_seed`: the warm
    start's and the closed loop's): dense rows of ``Z = L^-1 J^T`` (:class:`.SparseFactor`, entry ``l`` at coordinate ``l``), lane ``l`` holding
    coordinate ``l`` of ``u`` and of the rows it acts on, which the schedule loads into registers (:func:`_load`, or
    the pipeline of :func:`_consume`).
    Under ``torsion`` a friction pair may have its contact's spin row after it (:func:`.sweep.spin_row`): such a
    contact is four rows, its normal and its friction triple visited one by one, the triple from its metric cached
    for the launch (:func:`.sweep.friction_metric_law`).

    The schedule visits the rows in slot order: row by row; with ``contact_units``, each row, and a row with the tangent pair it owns (patch contacts own one when they hold an
    anchor) as a contact unit (:func:`_contact_unit`: a loaded unit on its Gram block, one ladder and one apply). The
    sweep owns every row of its worlds, its free body's among them: the free body's contacts are rows of the line, and
    with ``speed_rows`` its speed rows are the line's last (from ``speed_first``), which each sweep's last word visits
    on the body's six lanes (:func:`.sweep.speed_rows_pass`, the body's walk read through
    :data:`.sweep_routes.FREE_BODY_COORDINATES`).

    The rows are the walk's (:mod:`.walk_rows`): the velocity change in the walk's coordinates goes to the step's one
    leave (:attr:`.factor.walk.WalkTrees.du`, :func:`.factor.walk.walk_leave_kernel`), which walks each component's
    tree down in the rows' frame.

    The world's lane group is the device's (:attr:`PairedSweep.lanes`): a warp on CUDA, or the CPU's one thread,
    which holds each register's every entry (:func:`_hold`).

    Returns the body and its staging.
    """
    M, D = int(sweep.max_constraints), int(sweep.max_world_dofs)
    # The row store's line (its dense rows, then the free bodies'), the packed rows' too.
    pitch = M if sweep.pitch is None else int(sweep.pitch)
    torsion, speed_rows, solve_policy = sweep.torsion, sweep.speed_rows, sweep.solve_policy
    lanes = sweep.lanes
    threads = lanes * _PAIRED_WORLDS_PER_BLOCK if lanes > 1 else 1
    lanes = Lanes(lanes)
    laws = sweep.laws
    op = _rows(D, lanes)
    # The world's rows staged whole; a step divides by the diagonal.
    rows = StagedRows(M, lanes, line="off", count="m", reciprocal=False)
    staging = Staging(
        # A speed row stages its body's six coordinates (sweep.speed_row).
        (*rows.buffers(), *((Buffer("s_speed", "float", 6),) if speed_rows else ())),
        groups=threads // lanes.width,
        threads=threads,
        name="PairedScratch",
    )
    # A friction triple reads its metric cached for the launch (sweep.friction_metric_law).
    cached = "ROW_TRIPLE" if torsion else None
    # The schedule's rows: the line's every row but its speed rows, its last.
    end = "m_rows" if speed_rows else "m"
    ctx = RowContext.of(rows, op, end=end, metric=cached, inline_velocity_limit=True)
    schedule = _paired_units(ctx, laws, end) if sweep.contact_units else _paired_rows(rows, op, laws, end, cached)
    last_word = ()
    world_rows = "if (m == 0) return;"
    if speed_rows:
        world_rows += "\n    const int m_rows = min(speed_first.data[world], m);"
        body = ((), "secondary_offset", f"{_ROWS}.data[row_base + i * {D} + secondary_offset + 3]")
        speed = speed_rows_pass(
            rows, op, laws, walk=FREE_BODY_COORDINATES, start="m_rows", end="m", body=body, pad="        "
        )
        last_word = (speed,)
    write = "global_dof >= 0"
    # Where spin rows may appear, every friction triple's metric is formed once per launch, at its first loaded visit
    # (sweep.friction_metric_stage), and its later visits only project.
    metric = ""
    if torsion:
        metric = (
            friction_metric_stage(
                *rows.scan(),
                ctx.spin_row,
                end="m",
                record=friction_metric,
                pad="    ",
            )
            + "\n"
            + friction_metric_law("ROW_TRIPLE", friction_metric, "    ")
            + "\n"
        )
    # The impulses the rows carry into the first sweep, row by row in slot order (the general sweep's seed).
    seed = ""
    if sweep.seeded:
        apply = op.axpy((("row", "seed"),), "        ")
        row = (_hold(op, "row_z", "i"),)
        seed = impulse_seed("m", rows.load("i"), rows.kind("i"), apply, op.lanes.sync(), declare=row) + "\n"
    velocity = op.velocity.array
    if lanes.parallel:
        held = f"""    // The velocity change in factor coordinates per component, coordinate LANE in this lane's register.
    const int global_dof = lane < {D} ? world_dof_indices.data[dof_map_base + lane] : -1;
    float {velocity} = 0.0f;"""
        leave = f"""
    // The velocity change in the walk's coordinates, each lane's coordinate, to the step's one leave
    // (tree_factor.walk_leave_kernel), which walks each component's tree down in the rows' frame.
    if ({write}) walk.du.data[global_dof] += {velocity};"""
    else:
        held = f"""    // The velocity change in factor coordinates per component, every coordinate in the one lane's registers.
    float {velocity}[{D}];
    for (int d = 0; d < {D}; ++d) {velocity}[d] = 0.0f;"""
        leave = f"""
    // The velocity change in the walk's coordinates to the step's one leave (tree_factor.walk_leave_kernel).
    for (int d = 0; d < {D}; ++d) {{
        const int global_dof = world_dof_indices.data[dof_map_base + d];
        if ({write}) walk.du.data[global_dof] += {velocity}[d];
    }}"""
    leave = rows.stores("    ") + leave
    body = f"""{staging.declarations("    ")}
    const int lane = LANE;
    int m = world_constraint_count.data[world];
    if (m > {M}) m = {M};
    {world_rows}

    const int line = dense_first.data[world];
    __builtin_assume(line >= 0);
    const int off = line * {pitch};
    const int row_base = line * {pitch * D};
    const int dof_map_base = world * {D};
    const int secondary_offset = secondary_offset_by_world.data[world];
{rows.rings("    ")}

{rows.loads("    ")}
{held}
    SYNC();
{seed}{metric}
{level_loop((indent(schedule, "        "),), last_word, exit=end_sweep(solve_policy, iteration=ITERATION))}

{leave}
{staging.undefine()}
"""
    cone = (CONE_LAW,) if torsion else ()
    preludes = (
        lane_group(lanes.width),
        SHARED_STORAGE,
        RING,
        *law_preludes(PROJECTION_LAW, *cone),
        *solve_policy.preludes,
        *((FREE_BODY_WALKS,) if speed_rows else ()),
    )
    return group_native(body, preludes=preludes, lanes=lanes.width), staging


def _paired_rows(rows: StagedRows, op: SparseFactor, laws: RowLaws, end: str, metric: str | None) -> str:
    """Rows ``[0, end)`` one at a time in slot order by the row step (:func:`.sweep.row_step`), velocity limits in
    place, each row's factor row pipelined; a friction pair at its first tangent, its sibling's factor row loaded with
    the pair."""
    sibling = _hold(op, "sibling_z", "parent + 2", f"row_type == {FRICTION}") + "\n"
    if laws.torsion:
        # The pair's contact's spin row if it has one (the visit's ``spin``), solved with it (the friction triple).
        sibling += _hold(op, "spin_z", "spin", f"row_type == {FRICTION} && spin >= 0") + "\n"
    # A warp's lanes load each row's entries a row ahead; the one lane loads its row at the visit.
    prologue = f"float pre_z = {_load(op, '0')};" if op.lanes.parallel else ""
    order = Pipelined(end, lambda pad: f"{pad}{prologue}", lambda row, pad: _consume(op, row, end, pad))
    step = row_step(rows, op, laws, end=end, loads=sibling, inline_velocity_limit=True, metric=metric, pad="    ")
    return order.wrap(step, "")


def _contact_unit(ctx: RowContext, laws: RowLaws, kind: str, mu: str) -> str:
    """The contact unit headed by row ``head`` (of kind ``kind``, never friction) with its tangent pair ``(head + 1,
    head + 2)`` (friction coefficient ``mu`` at the visited row ``i``), its three factor rows in registers, by the unit
    law (:func:`.sweep.contact_unit`).

    The unit's impulses choose its reductions, not its law. A loaded unit (a nonzero impulse among its rows) reduces its
    rows' velocity terms, the pair's couplings with the normal and their terms' magnitudes in one ladder, runs its rows
    on :class:`.sweep.UnitZ` on the impulses its test loaded and applies their changes in one pass, after which the
    lanes synchronize once (:func:`.sweep.contact_unit`). An unloaded one visits its rows as the row-by-row
    sweep does: the normal on its own ladder (:func:`_unit_row`), then its pair on the pair's own (:func:`_pair_unit`),
    only on a nonzero disk, so a unit that stays inactive reduces the normal's terms alone.

    Under torsion a contact with a spin row (``head + 3``, :func:`.sweep.spin_row`) declines the unit: its normal and
    then its friction triple take their own visits, the triple from its metric cached for the launch
    (:func:`.sweep.friction_metric_law`), as on every sweep that solves a contact's rows one by one.
    """
    op = ctx.operator
    sums = (
        ("unit_jv0", "row", None),
        ("unit_jv1", "tangent1", None),
        ("unit_jv2", "tangent2", None),
        ("gram_n1", "row", "tangent1"),
        ("gram_n2", "row", "tangent2"),
        ("unit_cross", "tangent1", "tangent2"),
    )
    unit_op = dataclasses.replace(
        op, factor=_registers(op.lanes, row="row_z", tangent1="tangent1_z", tangent2="tangent2_z")
    )
    unit_ladder = unit_op.dot(
        sums, {"row": "head", "tangent1": "head + 1", "tangent2": "head + 2"}, "    ", magnitudes=("gram_n1", "gram_n2")
    )
    rows = dataclasses.replace(ctx, kind="unit_type", mu=lambda row: mu, inline_velocity_limit=False)
    declare = (
        f"const int i = head + r;\nconst int unit_type = r == 0 ? {kind} : {FRICTION};\nconst float denom = s_diag[i];"
    )
    # The unit is a normal and its pair: its rows' laws see no spin row.
    pair_laws = dataclasses.replace(laws, torsion=False)
    # The unit's ladder also reduces the pair's coupling and the rows' terms' magnitudes, so a moved tangent votes on a
    # lower bound of its terms that is the row-by-row sweep's own when the normal did not move.
    unit_ctx = dataclasses.replace(rows, operator=UnitZ(magnitudes=True))
    unit_rows = contact_unit(
        unit_ctx, pair_laws, rows="3", declare=declare, parent="head", check="denom", held=True, pad="    "
    )

    impulses, *staged = UNIT_HELD
    lines = [
        _hold(op, "row_z", "head"),
        _hold(op, "tangent1_z", "head + 1"),
        _hold(op, "tangent2_z", "head + 2"),
        unit_held_loads(unit_ctx, impulses, "head", ""),
        "if (unit_lam0 != 0.0f || unit_lam1 != 0.0f || unit_lam2 != 0.0f) {",
        *(unit_held_loads(unit_ctx, name, "head", "    ") for name in staged),
        "    // The unit's velocity terms, the pair's Gram couplings with the normal and their magnitudes, in one "
        "ladder.",
        unit_ladder,
        unit_rows,
        "    // The unit's changes in one pass: u += (Z_n dn + Z_t1 dt1) + Z_t2 dt2; then its stores are every lane's.",
        "    if (unit_dn != 0.0f || unit_dt1 != 0.0f || unit_dt2 != 0.0f) {",
        unit_op.axpy((("row", "unit_dn"), ("tangent1", "unit_dt1"), ("tangent2", "unit_dt2")), "        ", block=True),
        "    }",
        f"    {ctx.lanes.sync()}",
        "} else {",
        "    const int i = head, tangent1 = head + 1;",
        indent(_unit_row(ctx, pair_laws, kind, load=False), "    "),
        indent(_pair_unit(ctx, pair_laws, mu), "    "),
        "}",
    ]
    three = "\n".join(lines)
    if not laws.torsion:
        return three
    # A contact with a spin row: the normal by its row law, then the triple (its pair and spin row) on its own ladder.
    triple = dataclasses.replace(ctx, triple_law="ROW_TRIPLE(i, ")
    spun = [
        "const int i = head, tangent1 = head + 1;",
        _unit_row(ctx, laws, kind),
        _hold(op, "tangent1_z", "head + 1"),
        _hold(op, "tangent2_z", "head + 2"),
        _hold(op, "spin_z", "head + 3"),
        _pair_unit(triple, laws, mu, spin=ctx.load("tangent1 + 2")),
    ]
    return (
        f"if ({ctx.spin_row('head')} >= 0) {{\n{indent(chr(10).join(spun), '    ')}\n}} else {{\n"
        f"{indent(three, '    ')}\n}}"
    )


def _pair_unit(ctx: RowContext, laws: RowLaws, mu: str, *, spin: str | None = None) -> str:
    """The tangent pair ``(tangent1, tangent1 + 1)`` of the normal row ``i`` (its rows in ``tangent1_z``,
    ``tangent2_z``), solved as a unit after it on its own ladder (:func:`.sweep.pair_update`); under torsion, for a
    contact with a spin row (``tangent1 + 2``, its impulse ``spin``), the friction triple."""
    op = ctx.operator
    pair = dataclasses.replace(
        ctx,
        row="tangent1",
        kind=str(FRICTION),
        operator=dataclasses.replace(
            op, factor=_registers(op.lanes, row="tangent1_z", sib="tangent2_z", spin="spin_z")
        ),
        diagonal=Diagonal(lambda row: f"s_diag[{row}]"),
        mu=lambda row: ctx.mu(row) if row == "spin" else mu,
        parent="i",
        inline_velocity_limit=False,
    )
    spun = {"load_spin": spin} if laws.torsion else {}
    lines = [
        "float radius = 0.0f, load = 0.0f;" if laws.torsion else "float radius = 0.0f;",
        pair_radius(ctx.ring_load("i"), mu, ctx.load("tangent1"), ctx.load("tangent1 + 1"), "", **spun),
        pair_update(pair, laws, pad=""),
    ]
    return "\n".join(lines)


def _unit_row(ctx: RowContext, laws: RowLaws, kind: str, *, load: bool = True) -> str:
    """Row ``i`` of a unit (its kind ``kind`` is never friction) by the row law, when its diagonal is positive; its
    factor row loaded here, or (without ``load``) already."""
    op = ctx.operator
    row = dataclasses.replace(
        ctx,
        kind=kind,
        diagonal=Diagonal(lambda row: f"s_diag[{row}]"),
        friction=False,
        inline_velocity_limit=kind == "row_type",
    )
    lines = [
        _hold(op, "row_z", "i") if load else "",
        "if (s_diag[i] > 0.0f) {",
        row_sums(op, "jv", "    "),
        row_update(row, laws, residual="jv", pad="    "),
        "}",
    ]
    return "\n".join(line for line in lines if line)


def _paired_units(ctx: RowContext, laws: RowLaws, end: str = "m") -> str:
    """Contact units in slot order: a row by the row law, or, with the tangent pair it owns, a contact unit.

    A row owns the next two rows when they are tangent rows naming it as parent (and the spin row after them where it
    has one: :func:`.sweep.spin_row`). A tangent row outside a unit is left alone.
    """
    lines = [
        "    const int row_type = static_cast<int>(s_type[i]);",
        "    step = 1;",
        *((f"    if (row_type == {TORSION}) continue;",) if laws.torsion else ()),
        f"    if (row_type == {FRICTION}) continue;",
        "    const int tangent1 = i + 1;",
        f"    const bool owns_tangents = i + 2 < {end} && static_cast<int>(s_type[tangent1]) == {FRICTION}",
        "        && world_row_parent.data[off + tangent1] == i;",
        "    if (!owns_tangents) {",
        indent(_unit_row(ctx, laws, "row_type"), "        "),
        "        continue;",
        "    }",
        f"    step = {ctx.spin_row('i')} >= 0 ? 4 : 3;" if laws.torsion else "    step = 3;",
        "    const int head = i;",
        indent(_contact_unit(ctx, laws, "row_type", "world_row_mu.data[off + tangent1]"), "    "),
    ]
    return UnitStride(end).wrap("\n".join(lines), "")


@dataclasses.dataclass(frozen=True)
class PairedSweep(SweepConfig):
    """The paired factor sweep's configuration (case K5): a world of an articulation of ``primary_dofs`` coordinates
    and a free body of ``secondary_dofs``, ``max_world_dofs`` in all, its rows (at most ``max_constraints``) in factor
    coordinates, ``u`` a coordinate a lane in registers (:func:`paired_factor_source`).

    Its switches, each the plan's from the class's shape and the laws (:meth:`.sweep_plan.Sweep.build`):

    - ``u``'s place: in registers, a coordinate a lane, since the world's coordinates fit the lanes (the class plan's
      paired rule);
    - the enumeration: rows in slot order, or with ``contact_units`` a contact's rows on its Gram block (no
      compliance);
    - ``seeded``: the first sweep starts from the response of the impulses the rows carry (the warm start, the
      closed loop);
    - ``speed_rows``: the free body's speed rows are the line's last, the last word of each sweep;
    - ``pitch``: the row store's line, by default ``max_constraints`` rows.

    A world is a warp on CUDA and one thread on the CPU (:meth:`work`); weighted rows' and material contacts'
    operands (:func:`.sweep.weight_arguments`, :func:`.sweep.material_arguments`), a readout's operand and the walk's
    view follow ``v_out``.
    """

    max_constraints: int
    max_world_dofs: int
    primary_dofs: int
    secondary_dofs: int
    device_arch: str
    _: dataclasses.KW_ONLY
    contact_units: bool = False
    pitch: int | None = None
    seeded: bool = False
    speed_rows: bool = False

    ABI: ClassVar[tuple] = PAIRED_FACTOR_ABI
    NATIVE: ClassVar[str] = "sweep_paired_factor_native"
    LABEL: ClassVar[str] = "the paired factor sweep"

    def __post_init__(self):
        M, D, P, S = self.max_constraints, self.max_world_dofs, self.primary_dofs, self.secondary_dofs
        if M <= 0:
            raise ValueError("max_constraints must be positive")
        if P <= 0 or S <= 0 or P + S != D or D > 32:
            raise ValueError("paired factor PGS requires two positive components spanning at most 32 world DOFs")

    @property
    def lanes(self) -> int:
        """A world's lanes: a warp on CUDA, the CPU's one thread (:func:`.execution.device_lanes`)."""
        return device_lanes(self.device_arch, 32)

    @property
    def name(self) -> str:
        return (
            f"sweep_paired_factor_{self.max_constraints}_{self.max_world_dofs}_{self.primary_dofs}_{self.secondary_dofs}"
            + ("_units" if self.contact_units else "")
            + ("_weighted" if self.weighted else "")
            + ("_material" if self.material else "")
            + ("_torsion" if self.torsion else "")
            + ("" if self.pitch is None or self.pitch == self.max_constraints else f"_p{int(self.pitch)}")
            + ("_seeded" if self.seeded else "")
            + ("_speed" if self.speed_rows else "")
            + "_walk"
            + ("" if self.lanes == 32 else f"_lanes{self.lanes}")
            + self.solve_policy.suffix
        )

    source = paired_factor_source

    def dispatch(self, staging: Staging) -> Dispatch:
        return kernel_template.items("world", groups=staging.groups, lanes=self.lanes, guard="world_count")

    def operands(self) -> tuple[RowArgument, ...]:
        return (
            # Where spin rows may appear, every friction triple's metric (sweep.friction_metric_stage).
            *((RowArgument("friction_metric", wp.array3d[float], after="omega"),) if self.torsion else ()),
            *weight_arguments(self.weighted),
            *material_arguments(self.material),
            *self.solve_policy.arguments,
            RowArgument("walk", WalkTrees),
        )

    def work(self, items: int = 0, shared: int = 0) -> Work:
        """A warp a world, :data:`_PAIRED_WORLDS_PER_BLOCK` a block (one thread a world on the CPU), at the register
        line :data:`PAIRED_REGISTER_LINE`, which bounds the kernel where its block's ``shared`` memory keeps no more
        blocks than the line."""
        return Work(
            int(items),
            lanes=32,
            block=32 * _PAIRED_WORLDS_PER_BLOCK,
            registers=PAIRED_REGISTER_LINE,
            cap=False,
            shared=int(shared),
        )

# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The paired factor sweep (case K5) as a configuration of the sweep emitter (:mod:`.sweep`).

:func:`paired_factor_kernel`: one warp per world of an articulation and a free body, rows in factor coordinates
(:class:`.response_operator.FactorRows`) on the velocity change from the predictor, which leaves per component, under
one of two schedules: row by row, or contact units. Its working set fits the static
budget of the device it is built for (:func:`.memory.arch_budget`).
"""

from __future__ import annotations

import dataclasses
from functools import cache

import warp as wp

from ...core.memory import Buffer, Staging, arch_budget
from . import kernel_template
from .device_preludes import RING, SHARED_STORAGE, cuda_native, lane_group
from .factor.walk import WalkTrees
from .kernel_template import KernelTemplate
from .law_source import law_preludes
from .projection_laws import CONE_LAW, PROJECTION_LAW
from .response_operator import FactorComponent, FactorRows, Lanes
from .row_inputs import RowArgument
from .solve_policy import DEFAULT_POLICY, SolvePolicy
from .sweep import (
    FRICTION,
    TORSION,
    UNIT_HELD,
    Diagonal,
    ReducedPairCoupling,
    RowContext,
    RowLaws,
    UnitOperator,
    contact_unit,
    end_sweep,
    friction_metric,
    friction_metric_law,
    friction_metric_stage,
    indent,
    material_arguments,
    pair_radius,
    pair_update,
    row_update,
    row_visit,
    spin_row,
    unit_held_loads,
    weight_arguments,
)

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
    ("world_mf_constraint_count", wp.array[int]),
    ("primary_offset_by_world", wp.array[int]),
    ("secondary_offset_by_world", wp.array[int]),
    ("iterations", int),
    ("omega", float),
    ("v_out", wp.array[float]),
)
"""The paired factor sweep's native ABI: one world per warp, two per block (a tail-guarded tiled dispatch)."""

_PAIRED_WORLDS_PER_BLOCK = 2


def paired_factor_source(
    max_constraints: int,
    max_world_dofs: int,
    primary_dofs: int,
    secondary_dofs: int,
    independent_components: bool = False,
    *,
    contact_units: bool = False,
    torsion: bool = False,
    solve_policy: SolvePolicy = DEFAULT_POLICY,
    weighted: bool = False,
    material: bool = False,
    pitch: int | None = None,
) -> tuple[str, Staging]:
    """The native body of the paired factor sweep: a world's rows in the factor coordinates of its articulation and
    free body (:class:`.FactorRows`), a warp per world, the velocity change from the predictor, ``u = L^T dv``, from
    zero in a register per lane.
    Under ``torsion`` a friction pair may have its contact's spin row after it (:func:`.sweep.spin_row`): such a
    contact is four rows, its normal and its friction triple visited one by one, the triple from its metric cached
    for the launch (:func:`.sweep.friction_metric_law`).

    The schedule visits the rows in slot order: row by row; with ``contact_units``, each row, and a row with the tangent pair it owns (patch contacts own one when they hold an
    anchor) as a contact unit (:func:`_contact_unit`: a loaded unit on its Gram block, one ladder and one apply). With
    ``independent_components`` a world whose signed free-body row count is negative solves only its articulation's
    coordinates (its free body's rows run in the matrix-free sweep).

    The rows are the walk's (:mod:`.walk_rows`): the velocity change in the walk's coordinates goes to the step's one
    leave (:attr:`.factor.walk.WalkTrees.du`, :func:`.factor.walk.walk_leave_kernel`), which walks each component's
    tree down in the rows' frame.

    Returns the body and its staging.
    """
    M, D, P, S = int(max_constraints), int(max_world_dofs), int(primary_dofs), int(secondary_dofs)
    # The row store's line (its dense rows, then the free bodies'); the packed rows keep M rows a world.
    pitch = M if pitch is None else int(pitch)
    if M <= 0:
        raise ValueError("max_constraints must be positive")
    if P <= 0 or S <= 0 or P + S != D or D > 32:
        raise ValueError("paired factor PGS requires two positive components spanning at most 32 world DOFs")
    lanes = Lanes(32)
    laws = RowLaws(regularize=weighted, material=material, torsion=torsion, policy=solve_policy)
    op = FactorRows(
        D,
        (
            FactorComponent(P, "primary_offset", "primary_lane"),
            FactorComponent(S, "secondary_offset", "secondary_lane"),
        ),
        lanes,
    )
    staging = Staging(
        (
            Buffer("s_lam", "float", M),
            Buffer("s_rhs", "float", M),
            Buffer("s_diag", "float", M),
            Buffer("s_type", "unsigned char", M),
        ),
        groups=_PAIRED_WORLDS_PER_BLOCK,
        threads=32 * _PAIRED_WORLDS_PER_BLOCK,
        name="PairedScratch",
    )
    ctx = RowContext(
        row="i",
        kind="row_type",
        operator=op,
        diagonal=Diagonal(lambda row: "denom" if row == "i" else f"s_diag[{row}]"),
        load=lambda row: f"s_lam[{row}]",
        store=lambda row, value: lanes.first(f"s_lam[{row}] = {value};"),
        rhs=lambda row: f"s_rhs[{row}]",
        mu=lambda row: f"world_row_mu.data[off + {row}]",
        weight=lambda row: f"world_row_w.data[off + {row}]",
        eps=lambda row: f"world_row_eps.data[off + {row}]",
        parent="parent",
        ring_load=lambda parent, load=None: (
            f"RING_LOAD({load or f's_lam[{parent}]'}, {parent}, m, ring_parent, ring_impulse)"
        ),
        lanes=lanes,
        inline_velocity_limit=True,
        spin_row=lambda parent: spin_row(parent, "m", lambda row: f"static_cast<int>(s_type[{row}])"),
        # A friction triple reads its metric cached for the launch (sweep.friction_metric_law).
        **({"triple_law": "ROW_TRIPLE(parent, "} if torsion else {}),
    )
    schedule = _paired_units(ctx, laws) if contact_units else _paired_rows(ctx, laws, "m")

    world_rows = "if (m == 0 || world_mf_constraint_count.data[world] != 0) return;"
    write = "global_dof >= 0"
    if independent_components:
        world_rows = (
            "if (m == 0 || world_mf_constraint_count.data[world] > 0) return;\n"
            "    const bool split_component = world_mf_constraint_count.data[world] < 0;"
        )
        write += f" && (!split_component || (primary_lane >= 0 && primary_lane < {P}))"
    # Where spin rows may appear, every friction triple's metric is formed once per launch, at its first loaded visit
    # (sweep.friction_metric_stage), and its later visits only project.
    metric = ""
    if torsion:
        metric = (
            friction_metric_stage(
                lambda row: f"static_cast<int>(s_type[{row}])",
                lambda row: f"world_row_parent.data[off + {row}]",
                ctx.spin_row,
                end="m",
                record=friction_metric,
                pad="    ",
            )
            + "\n"
            + friction_metric_law("ROW_TRIPLE", friction_metric, "    ")
            + "\n"
        )
    leave = f"""    for (int i = lane; i < m; i += 32) world_impulses.data[off + i] = s_lam[i];
    // The velocity change in the walk's coordinates, each lane's coordinate, to the step's one leave
    // (tree_factor.walk_leave_kernel), which walks each component's tree down in the rows' frame.
    if ({write}) walk.du.data[global_dof] += {op.velocity};"""
    body = f"""{staging.declarations("    ")}
    const int lane = LANE;
    int m = world_constraint_count.data[world];
    if (m > {M}) m = {M};
    {world_rows}

    const int line = dense_first.data[world];
    __builtin_assume(line >= 0);
    const int off = line * {pitch};
    const int row_base = line * {M * D};
    const int dof_map_base = world * {D};
    const int primary_offset = primary_offset_by_world.data[world];
    const int secondary_offset = secondary_offset_by_world.data[world];
    // A contact normal's patch ring links the region's next normal (-1 ends a point contact).
    const auto ring_parent = [&](int r) {{ return world_row_parent.data[off + r]; }};
    const auto ring_impulse = [&](int r) {{ return s_lam[r]; }};

    for (int i = lane; i < m; i += 32) {{
        s_lam[i] = world_impulses.data[off + i];
        s_rhs[i] = rhs_bias.data[off + i] + row_incident.data[off + i];
        s_diag[i] = world_diag.data[off + i];
        s_type[i] = static_cast<unsigned char>(world_row_type.data[off + i]);
    }}
    // The velocity change in factor coordinates per component, coordinate LANE in this lane's register.
    const int global_dof = lane < {D} ? world_dof_indices.data[dof_map_base + lane] : -1;
{chr(10).join(op.lanes_of("    "))}
    float {op.velocity} = 0.0f;
    SYNC();
{metric}
    for (int iter = 0; iter < iterations; ++iter) {{
        int iteration_changed = 0;
{indent(schedule, "        ")}
{end_sweep(solve_policy, iteration="iter")}    }}

{leave}
{staging.undefine()}
#endif
"""
    cone = (CONE_LAW,) if torsion else ()
    preludes = (
        lane_group(32),
        SHARED_STORAGE,
        RING,
        *law_preludes(PROJECTION_LAW, *cone),
        *solve_policy.preludes,
    )
    return cuda_native(body, preludes=preludes), staging


def _paired_rows(ctx: RowContext, laws: RowLaws, end: str) -> str:
    """Rows ``[0, end)`` one at a time in slot order, each row's factor row pipelined; a friction pair at its first
    tangent, its sibling's factor row loaded with the pair."""
    op = ctx.operator
    sibling = f"const float {op.sibling} = row_type == {FRICTION} ? ({op.load('parent + 2')}) : 0.0f;\n"
    if laws.torsion:
        # The pair's contact's spin row if it has one (the visit's ``spin``), solved with it (the friction triple).
        sibling += f"const float {op.spin} = row_type == {FRICTION} && spin >= 0 ? ({op.load('spin')}) : 0.0f;\n"
    lines = [
        op.prefetch("0", ""),
        f"for (int i = 0; i < {end}; ++i) {{",
        op.consume("i", end, "    "),
        "    const int row_type = static_cast<int>(s_type[i]);",
        f"    if (row_type == {TORSION}) continue;" if laws.torsion else "",
        row_visit(
            ctx,
            laws,
            parent="world_row_parent.data[off + i]",
            check="denom",
            declare=sibling + "const float denom = s_diag[i];",
            pad="    ",
        ),
        "}",
    ]
    return "\n".join(line for line in lines if line)


@dataclasses.dataclass(frozen=True)
class _UnitFactorRows(UnitOperator, ReducedPairCoupling, FactorRows):
    """A contact unit's factor rows (:class:`.sweep.UnitOperator`): the unit's ladder (:func:`_contact_unit`) also
    reduces the pair's coupling and the rows' terms' magnitudes, so a moved tangent votes on a lower bound of its terms
    that is the row-by-row sweep's own when the normal did not move."""

    magnitudes = True


def _contact_unit(ctx: RowContext, laws: RowLaws, kind: str, mu: str) -> str:
    """The contact unit headed by row ``head`` (of kind ``kind``, never friction) with its tangent pair ``(head + 1,
    head + 2)`` (friction coefficient ``mu`` at the visited row ``i``), its three factor rows in registers, by the unit
    law (:func:`.sweep.contact_unit`).

    The unit's impulses choose its reductions, not its law. A loaded unit (a nonzero impulse among its rows) reduces its
    rows' velocity terms, the pair's couplings with the normal and their terms' magnitudes in one ladder, runs its rows
    on :class:`_UnitFactorRows` on the impulses its test loaded and applies their changes in one pass, after which the
    lanes synchronize once (:func:`.sweep.contact_unit`). An unloaded one visits its rows as the row-by-row
    sweep does: the normal on its own ladder (:func:`_unit_row`), then its pair on the pair's own (:func:`_pair_unit`),
    only on a nonzero disk, so a unit that stays inactive reduces the normal's terms alone.

    Under torsion a contact with a spin row (``head + 3``, :func:`.sweep.spin_row`) declines the unit: its normal and
    then its friction triple take their own visits, the triple from its metric cached for the launch
    (:func:`.sweep.friction_metric_law`), as on every sweep that solves a contact's rows one by one.
    """
    op = ctx.operator
    velocity = op.velocity
    unit = _UnitFactorRows(**{field.name: getattr(op, field.name) for field in dataclasses.fields(op)})
    channels = (
        ("unit_jv0", "row_z", velocity),
        ("unit_jv1", "tangent1_z", velocity),
        ("unit_jv2", "tangent2_z", velocity),
        ("gram_n1", "row_z", "tangent1_z"),
        ("gram_n2", "row_z", "tangent2_z"),
        ("unit_cross", "tangent1_z", "tangent2_z"),
    )
    rows = dataclasses.replace(ctx, kind="unit_type", mu=lambda row: mu, inline_velocity_limit=False)
    declare = (
        f"const int i = head + r;\nconst int unit_type = r == 0 ? {kind} : {FRICTION};\nconst float denom = s_diag[i];"
    )
    # The unit is a normal and its pair: its rows' laws see no spin row.
    pair_laws = dataclasses.replace(laws, torsion=False)
    unit_ctx = dataclasses.replace(rows, operator=unit)
    unit_rows = contact_unit(
        unit_ctx, pair_laws, rows="3", declare=declare, parent="head", check="denom", held=True, pad="    "
    )

    impulses, *staged = UNIT_HELD
    lines = [
        f"const float row_z = {op.load('head')};",
        f"const float tangent1_z = {op.load('head + 1')};",
        f"const float tangent2_z = {op.load('head + 2')};",
        unit_held_loads(unit_ctx, impulses, "head", ""),
        "if (unit_lam0 != 0.0f || unit_lam1 != 0.0f || unit_lam2 != 0.0f) {",
        *(unit_held_loads(unit_ctx, name, "head", "    ") for name in staged),
        "    // The unit's velocity terms, the pair's Gram couplings with the normal and their magnitudes, in one "
        "ladder.",
        unit._sums(channels, "    ", magnitudes=("gram_n1", "gram_n2")),
        unit_rows,
        "    // The unit's changes in one pass: u += Z_n dn + Z_t1 dt1 + Z_t2 dt2; then its stores are every lane's.",
        "    if (unit_dn != 0.0f || unit_dt1 != 0.0f || unit_dt2 != 0.0f)",
        f"        {velocity} += (row_z * unit_dn + tangent1_z * unit_dt1) + tangent2_z * unit_dt2;",
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
        f"const float tangent1_z = {op.load('head + 1')};",
        f"const float tangent2_z = {op.load('head + 2')};",
        f"const float {op.spin} = {op.load('head + 3')};",
        _pair_unit(triple, laws, mu, spin="s_lam[tangent1 + 2]"),
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
        operator=dataclasses.replace(op, current="tangent1_z", sibling="tangent2_z"),
        diagonal=Diagonal(lambda row: f"s_diag[{row}]"),
        mu=lambda row: ctx.mu(row) if row == "spin" else mu,
        parent="i",
        inline_velocity_limit=False,
    )
    spun = {"load_spin": spin} if laws.torsion else {}
    lines = [
        "float radius = 0.0f, load = 0.0f;" if laws.torsion else "float radius = 0.0f;",
        pair_radius(ctx.ring_load("i"), mu, "s_lam[tangent1]", "s_lam[tangent1 + 1]", "", **spun),
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
        f"const float {op.current} = {op.load('i')};" if load else "",
        "if (s_diag[i] > 0.0f) {",
        op.residual("jv", "    "),
        row_update(row, laws, residual="jv", pad="    "),
        "}",
    ]
    return "\n".join(line for line in lines if line)


def _paired_units(ctx: RowContext, laws: RowLaws) -> str:
    """Contact units in slot order: a row by the row law, or, with the tangent pair it owns, a contact unit.

    A row owns the next two rows when they are tangent rows naming it as parent (and the spin row after them where it
    has one: :func:`.sweep.spin_row`). A tangent row outside a unit is left alone.
    """
    lines = [
        "for (int i = 0, step = 1; i < m; i += step) {",
        "    const int row_type = static_cast<int>(s_type[i]);",
        "    step = 1;",
        *((f"    if (row_type == {TORSION}) continue;",) if laws.torsion else ()),
        f"    if (row_type == {FRICTION}) continue;",
        "    const int tangent1 = i + 1;",
        f"    const bool owns_tangents = i + 2 < m && static_cast<int>(s_type[tangent1]) == {FRICTION}",
        "        && world_row_parent.data[off + tangent1] == i;",
        "    if (!owns_tangents) {",
        indent(_unit_row(ctx, laws, "row_type"), "        "),
        "        continue;",
        "    }",
        f"    step = {ctx.spin_row('i')} >= 0 ? 4 : 3;" if laws.torsion else "    step = 3;",
        "    const int head = i;",
        indent(_contact_unit(ctx, laws, "row_type", "world_row_mu.data[off + tangent1]"), "    "),
        "}",
    ]
    return "\n".join(lines)


@cache
def paired_factor_kernel(
    max_constraints: int,
    max_world_dofs: int,
    primary_dofs: int,
    secondary_dofs: int,
    device_arch: str,
    independent_components: bool = False,
    *,
    contact_units: bool = False,
    torsion: bool = False,
    solve_policy: SolvePolicy = DEFAULT_POLICY,
    weighted: bool = False,
    material: bool = False,
    pitch: int | None = None,
) -> wp.Kernel:
    """The paired factor sweep kernel; weighted rows' and material contacts' operands (:func:`.sweep.weight_arguments`,
    :func:`.sweep.material_arguments`), a readout's operand and the walk's view follow ``v_out``."""
    snippet, staging = paired_factor_source(
        max_constraints,
        max_world_dofs,
        primary_dofs,
        secondary_dofs,
        independent_components,
        contact_units=contact_units,
        torsion=torsion,
        solve_policy=solve_policy,
        weighted=weighted,
        material=material,
        pitch=pitch,
    )
    staging.check("the paired factor sweep", arch_budget(device_arch))
    name = (
        f"sweep_paired_factor_{max_constraints}_{max_world_dofs}_{primary_dofs}_{secondary_dofs}"
        + ("_units" if contact_units else "")
        + ("_components" if independent_components else "")
        + ("_weighted" if weighted else "")
        + ("_material" if material else "")
        + ("_torsion" if torsion else "")
        + ("" if pitch is None or pitch == max_constraints else f"_p{int(pitch)}")
        + "_walk"
        + solve_policy.suffix
    )
    walked = (RowArgument("walk", WalkTrees),)
    return KernelTemplate.from_abi(
        name,
        PAIRED_FACTOR_ABI,
        snippet,
        kernel_template.items("world", groups=_PAIRED_WORLDS_PER_BLOCK, lanes=32, guard="world_count"),
        extra=(
            # Where spin rows may appear, every friction triple's metric (sweep.friction_metric_stage).
            *((RowArgument("friction_metric", wp.array3d[float], after="omega"),) if torsion else ()),
            *weight_arguments(weighted),
            *material_arguments(material),
            *solve_policy.arguments,
            *walked,
        ),
        composed=True,
        native_name="sweep_paired_factor_native",
    ).kernel()

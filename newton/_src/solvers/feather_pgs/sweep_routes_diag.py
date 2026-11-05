# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The sparse-diagonal sweep as a configuration of the sweep emitter (:mod:`.sweep`).

:func:`sparse_diagonal_kernel`: a world's dense group plus diagonal-mass coordinates as whitened rows
(:class:`.SparsePairZ`), one warp per world, on the velocity change from the predictor in factor coordinates, from zero:
``du = R^-T L^T P dv`` on the dense articulation (the walk's factor and its slips' release, the frame its rows were whitened
in) and ``du = sqrt(m) dv`` on a diagonal coordinate, left once and added to the predictor. Each sweep runs its coordinates' fused joint limits, its independent scalar contacts one lane per chain, then its
coupled rows in order (plain or contact units, optionally with speculative batches), a loaded coupled
contact unit by the unit law (:meth:`.SparsePairZ.unit_terms`, :func:`.sweep.contact_unit`). Its working set fits the static
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
from .response_operator import Lanes, SparsePairZ
from .row_inputs import RowArgument
from .row_metadata import _validate_dense_metadata_encoding
from .solve_policy import DEFAULT_POLICY, Change, SolvePolicy, bound
from .sweep import (
    CONTACT,
    FRICTION,
    JOINT_LIMIT,
    TORSION,
    Diagonal,
    RowContext,
    RowLaws,
    UnitZ,
    WorldRows,
    change_vote,
    contact_unit,
    end_sweep,
    friction_metric,
    friction_metric_law,
    friction_metric_stage,
    indent,
    material_arguments,
    row_visit,
    weight_arguments,
)

# ---------------------------------------------------------------------------------------------------------------------
# Sparse diagonal: a dense group plus two scalar coordinates per row
# ---------------------------------------------------------------------------------------------------------------------

SPARSE_DIAGONAL_ABI = kernel_template.params(
    ("world_constraint_count", wp.array[int]),
    ("dense_first", wp.array[int]),
    ("world_dof_indices", wp.array2d[int]),
    ("rhs_bias", wp.array2d[float]),
    ("row_incident", wp.array2d[float]),
    ("world_diag", wp.array2d[float]),
    ("world_impulses", wp.array2d[float]),
    ("world_row_type", wp.array2d[int]),
    ("world_row_parent", wp.array2d[int]),
    ("world_row_mu", wp.array2d[float]),
    ("dense_phase_bounds", wp.array2d[int]),
    ("sparse_contact_group_count", wp.array[int]),
    ("sparse_contact_group_heads", wp.array2d[int]),
    ("sparse_contact_serial_count", wp.array[int]),
    ("sparse_contact_serial_normals", wp.array2d[int]),
    ("dense_offsets", wp.array[int]),
    ("dense_groups", wp.array[int]),
    ("walk", WalkTrees),
    ("dense_Z", wp.array3d[float]),
    ("sparse_row_dof", wp.array3d[int]),
    ("sparse_row_z", wp.array3d[float]),
    ("fused_limit_active", wp.array2d[int]),
    ("fused_limit_lower_rhs", wp.array2d[float]),
    ("fused_limit_upper_rhs", wp.array2d[float]),
    ("diagonal_inverse_mass", wp.array[float]),
    ("fused_limit_cfm", float),
    ("iterations", int),
    ("omega", float),
    ("fused_limit_lower_lambda", wp.array2d[float]),
    ("fused_limit_upper_lambda", wp.array2d[float]),
    ("v_out", wp.array[float]),
)
"""The sparse-diagonal sweep's operands (one world per launch item; a readout follows ``v_out``)."""


def _diagonal_limits(dofs: int, dense_dofs: int, laws: RowLaws) -> tuple[str, str, str, str]:
    """The fused joint limits of a world's diagonal-mass coordinates: declarations, loads, the pass and the stores.

    Coordinate ``d`` has a lower and an upper unilateral row (``J = +1``, ``-1``), unstored whitened rows on the
    coordinate: ``Z = +-s`` with ``s = m^-1/2`` (``limit_scale``), their right-hand sides holding their incident
    velocities ``+-v_hat`` (:func:`.row_kernels.prepare_fused_diagonal_joint_limits`), so ``J dv = +-s u``, ``A = s^2 + cfm`` (the step
    divides by it) and ``u += +-s delta``. The lane that owns ``d = lane + 32 k`` keeps both impulses in registers for
    the launch; they start at zero and are written back at the end. The dense articulation's ``dense_dofs``
    coordinates (from ``dense_offset``) hold no fused limit.
    """
    declarations, loads, rows, stores = [], [], [], []
    for k in range((dofs + 31) // 32):
        d = f"lane + {32 * k}" if k else "lane"
        declarations.append(
            f"    int limit_active_{k} = 0;\n"
            f"    float limit_lower_rhs_{k} = 0.0f, limit_upper_rhs_{k} = 0.0f, limit_scale_{k} = 0.0f;\n"
            f"    float limit_lower_lambda_{k} = 0.0f, limit_upper_lambda_{k} = 0.0f;"
        )
        loads.append(
            f"    if ({d} < {dofs}) {{\n"
            f"        const int limit_index = world * {dofs} + {d};\n"
            f"        const int global_dof = world_dof_indices.data[dof_map_base + {d}];\n"
            f"        limit_active_{k} = fused_limit_active.data[limit_index];\n"
            f"        limit_lower_rhs_{k} = fused_limit_lower_rhs.data[limit_index];\n"
            f"        limit_upper_rhs_{k} = fused_limit_upper_rhs.data[limit_index];\n"
            f"        const bool limit_dense = {d} >= dense_offset && {d} < dense_offset + {dense_dofs};\n"
            f"        if (global_dof >= 0 && !limit_dense) limit_scale_{k} = sqrtf(diagonal_inverse_mass.data[global_dof]);\n"
            "    }"
        )
        sides = []
        for bit, side, sign, update in ((1, "lower", "", "+="), (2, "upper", "-", "-=")):
            limit_bound = bound(f"limit_{side}_rhs_{k}", "fabsf(limit_velocity)", "old_lambda", diagonal="limit_denom")
            limit = Change("delta_lambda", "limit_bound", str(JOINT_LIMIT), "limit_denom", scaled=True)
            change = change_vote(laws, (limit,), " " * 20)
            sides.append(
                f"            if ((limit_active_{k} & {bit}) != 0) {{\n"
                f"                const float limit_velocity = limit_scale_{k} * s_v[{d}];\n"
                f"                const float residual = {sign}limit_velocity + limit_{side}_rhs_{k};\n"
                f"                const float old_lambda = limit_{side}_lambda_{k};\n"
                f"                const float limit_bound = {limit_bound};\n"
                "                const float new_lambda = ProjectionLaw::unilateral(\n"
                "                    ProjectionLaw::relaxed_step(old_lambda, -residual / limit_denom, omega));\n"
                "                const float delta_lambda = new_lambda - old_lambda;\n"
                f"                limit_{side}_lambda_{k} = new_lambda;\n"
                "                if (delta_lambda != 0.0f) {\n"
                f"{change}\n"
                f"                    s_v[{d}] {update} limit_scale_{k} * delta_lambda;\n"
                "                }\n"
                "            }"
            )
        rows.append(
            f"        if ({d} < {dofs} && limit_scale_{k} > 0.0f) {{\n"
            f"            const float limit_denom = limit_scale_{k} * limit_scale_{k} + fused_limit_cfm;\n"
            + "\n".join(sides)
            + "\n        }"
        )
        stores.append(
            f"    if ({d} < {dofs}) {{\n"
            f"        const int limit_index = world * {dofs} + {d};\n"
            f"        fused_limit_lower_lambda.data[limit_index] = limit_lower_lambda_{k};\n"
            f"        fused_limit_upper_lambda.data[limit_index] = limit_upper_lambda_{k};\n"
            "    }"
        )
    return "\n".join(declarations), "\n".join(loads), "\n".join(rows), "\n".join(stores)


def sparse_diagonal_source(
    max_constraints: int,
    max_world_dofs: int,
    dense_dofs: int,
    *,
    speculative_contact_batches: bool = False,
    contact_units: bool = False,
    torsion: bool = False,
    solve_policy: SolvePolicy = DEFAULT_POLICY,
    weighted: bool = False,
    material: bool = False,
) -> tuple[str, Staging]:
    """The native body of the sparse-diagonal Gauss-Seidel sweep: one warp per world.

    Every sweep runs the world's fused diagonal joint limits (one lane per coordinate), then its independent scalar
    contacts (one lane per chain: a contact whose coordinate no coupled row touches commutes with the coupled rows,
    and its one-product residual rounds as the warp's sum of it does), then the coupled rows in row order with
    :class:`.SparsePairZ`'s warp layout, all on the velocity change in factor coordinates, from zero (left once before
    it is added to the predictor). ``contact_units`` schedules the contacts (a normal, and its
    tangent pair only when it owns an anchor): it skips an independent contact in the serial order and a pair whose patch carries no load. Every schedule, the plain
    one at a normal row that owns the tangent pair after it, solves a loaded coupled contact unit by the unit law
    (:func:`.sweep.contact_unit`) and visits an unloaded one, and a chain's contacts, row by row: a chain runs one lane
    per contact, whose row visits reduce nothing, and a unit branch there splits the warp's lanes. With
    ``speculative_contact_batches`` the coupled contacts come from the world's serial list, and a batch of them
    (one per segment of the warp) is tested first: a batch whose normals stay zero without load is skipped whole,
    otherwise the sweep enters the batch at its first active contact. Under contact ``torsion`` a friction contact's
    spin row follows its pair (four rows: a triple's stride is four); such a contact declines the unit, and its pair's
    visit solves the triple from its metric cached for the launch (:func:`.sweep.friction_metric_law`). The dense
    articulation's change goes to the step's one leave in the walk's coordinates
    (:func:`.factor.walk.walk_leave_kernel`).

    Returns the body and its staging.
    """
    M, D, P = int(max_constraints), int(max_world_dofs), int(dense_dofs)
    if speculative_contact_batches and (not contact_units or P + 2 > 16):
        raise ValueError("speculative contact batches require contact units with at most 16 response entries")
    _validate_dense_metadata_encoding(M)
    batch_lanes = 8 if P + 2 <= 8 else 16
    batch_shift = batch_lanes.bit_length() - 1
    batch_contacts = 32 // batch_lanes
    scheduled = contact_units

    warp = Lanes(32)
    laws = RowLaws(regularize=weighted, material=material, policy=solve_policy, torsion=torsion)
    rows = WorldRows(M, lanes=warp)
    staging = Staging((Buffer("s_v", "float", D), *rows.buffers()), threads=32)

    def context(lanes: Lanes, *, friction: bool = True) -> RowContext:
        """The rows' context in a layout at row ``i``: the warp's (every lane stores, the warp synchronizes after the
        apply) or one thread's (an independent scalar contact on ``scalar_coord``)."""
        return RowContext(
            row="i",
            kind="row_type",
            operator=SparsePairZ(P, M, lanes),
            diagonal=Diagonal(lambda row: "denom" if row == "i" else rows.diag(row)),
            load=rows.peek,
            store=lambda row, value: f"{rows.peek(row)} = {value};",
            rhs=rows.rhs_of,
            mu=rows.mu,
            weight=rows.weight,
            eps=rows.eps,
            parent="parent",
            ring_load=rows.ring_load,
            lanes=lanes,
            inline_velocity_limit=False,
            # Every lane (or the one thread) stores what it computed; a pair finishes its second tangent in its branch,
            # on the entries it kept (SparsePairZ.pair_entries).
            owned=False,
            friction=friction,
            spin_row=lambda parent: rows.spin_of(parent, "row_count"),
            # A friction triple reads its metric cached for the launch (sweep.friction_metric_law).
            **({"triple_law": "ROW_TRIPLE(parent, "} if torsion else {}),
        )

    def visit(lanes: Lanes, pad: str, *, kind: int | None = None, parent: str | None = None) -> str:
        """Row ``i``'s visit in a layout (:func:`context`). A ``kind`` (and a friction row's ``parent``) known where the
        visit is emitted replaces the row's metadata, so the visit compiles to that kind's law; a known kind other than
        friction admits no pair."""
        ctx = context(lanes, friction=kind in (None, FRICTION))
        # A spin row is solved with its friction pair.
        skip = f"{pad}if (row_type == {TORSION}) continue;\n" if torsion and kind is None else ""
        return (
            f"{pad}const int row_type = {rows.kind('i') if kind is None else kind};\n"
            + skip
            + row_visit(
                ctx,
                laws,
                parent=rows.parent("i") if parent is None else parent,
                check="denom",
                declare=f"const float denom = {rows.diag('i')};",
                entries=ctx.operator.consume("i", ""),
                pad=pad,
            )
        )

    pair_op = SparsePairZ(P, M, warp)

    def contact_rows(head: str) -> str:
        """The rows of the friction contact headed by row ``head``: three, or four with a spin row (the data's,
        :func:`.sweep.spin_row`)."""
        return f"({rows.spin_of(head, 'row_count')} >= 0 ? 4 : 3)" if torsion else "3"

    # A contact unit is a normal and its pair: a contact with a spin row declines it, and its rows take their visits,
    # the triple from its cached metric.
    unit_rows = contact_unit(
        dataclasses.replace(context(warp), operator=UnitZ()),
        dataclasses.replace(laws, torsion=False),
        rows="unit_rows",
        declare=f"const int i = head + r;\nconst int row_type = r == 0 ? {CONTACT} : {FRICTION};\n"
        f"const float denom = {rows.diag('i')};",
        parent="head",
        check="denom",
        pad="    ",
    )
    loaded = " || ".join(f"{rows.peek(row)} != 0.0f" for row in ("i", "i + 1", "i + 2"))

    def coupled(condition: str, advance: str = "i") -> str:
        """At row ``i``, when ``condition`` holds (``i`` heads a coupled contact unit with a pair, and no spin row) and
        the unit is loaded, the unit on its Gram block in the warp's layout (:meth:`.SparsePairZ.unit_terms`); the loop then moves
        ``advance`` past its rows. An unloaded unit, or a contact with a spin row, falls through to the loop's row
        visits."""
        spinless = f" && {rows.spin_of('i', 'row_count')} < 0" if torsion else ""
        text = f"""if ({condition}{spinless}) {{
    const int unit_rows = 3;
    if ({loaded}) {{
        const int head = i;
{pair_op.unit_terms("head", "        ")}
{indent(unit_rows, "    ")}
{pair_op.unit_apply("        ")}
        {advance} += unit_rows - 1;
        continue;
    }}
}}"""
        return indent(text, " " * 12)

    def independent(head: str) -> str:
        """Whether the contact headed by row ``head`` belongs to an independent chain (a slot holds a link)."""
        return f"sparse_row_dof.data[(row_base + {head}) * 2] < -1 || sparse_row_dof.data[(row_base + {head}) * 2 + 1] < -1"

    def owns_tangents(head: str, pad: str) -> str:
        """Whether unit head ``head`` owns a tangent pair: the row after it is a tangent naming it as parent."""
        return (
            f"{head} + 2 < row_count\n"
            f"{pad}&& ({rows.kind(f'{head} + 1')}) == {FRICTION}\n"
            f"{pad}&& {rows.parent(f'{head} + 1', enclosed=True)} == {head}"
        )

    def inactive_pair(condition: str, advance: str) -> str:
        """Skip a contact's tangent rows (and spin row) when they hold nothing and its patch carries no load."""
        empty = "".join(f"\n                && {rows.peek(f'i + {r}')} == 0.0f" for r in range(1, 3))
        if torsion:
            empty += f"\n                && (unit_rows < 4 || {rows.peek('i + 3')} == 0.0f)"
        return (
            f"            if ({condition}) {{\n"
            f"                const int unit_rows = {contact_rows('i')};\n"
            f"                if (true{empty}) {{\n"
            f"                    const float patch_load = {indent(rows.ring_load('i'), '                    ').lstrip()};\n"
            f"                    if (patch_load <= 0.0f) {advance} += unit_rows - 1;\n"
            "                }\n"
            "            }"
        )

    # Independent scalar chains: one lane per chain, its contacts in chain order. Schedule construction excludes every
    # coordinate a coupled row touches, so they commute with the coupled rows. A contact's normal links the chain's
    # next contact through its free slot (-2 ends the chain). A contact is a normal row and, for a triple or a unit
    # that owns an anchor, the tangent pair after it, which names the normal as parent: the visits' kinds are known
    # here, so each compiles to its own law. The pair is solved at its first tangent. A visit that skips its row leaves
    # its block (do-while), not the chain.
    independent_phase = ""
    if scheduled:
        pad = " " * 16
        tangents = f"""{pad}do {{
{pad}    const int i = head + 1;
{visit(Lanes(1), pad + "    ", kind=FRICTION, parent="head")}
{pad}}} while (false);"""
        tangents = f"{pad}if ({owns_tangents('head', pad + '    ')}) {{\n{indent(tangents, '    ')}\n{pad}}}"
        independent_phase = f"""
        const int independent_group_count = sparse_contact_group_count.data[world];
        for (int group = lane; group < independent_group_count; group += 32) {{
            int next_normal = sparse_contact_group_heads.data[world * {D} + group];
            while (next_normal >= 0) {{
                const int head = next_normal;
                const int normal_coord_0 = sparse_row_dof.data[(row_base + head) * 2];
                const int normal_coord_1 = sparse_row_dof.data[(row_base + head) * 2 + 1];
                const int scalar_coord = normal_coord_0 >= 0 ? normal_coord_0 : normal_coord_1;
                const int link = normal_coord_0 < -1 ? normal_coord_0 : normal_coord_1;
                next_normal = link == -2 ? -1 : -link - 3;
                do {{
                    const int i = head;
{visit(Lanes(1), pad + "    ", kind=CONTACT)}
                }} while (false);
{tangents}
            }}
        }}
        SYNC();
"""

    def batch(index: str, count: str) -> str:
        """Test the batch of serial contacts from ``index``: one contact per segment of ``batch_lanes`` lanes, its
        normal's prospective update; declares ``noop_lanes``, the lanes whose contact would change nothing."""
        segment = SparsePairZ(P, M, Lanes(batch_lanes, "batch_lane", segment=True))
        step = Diagonal(rows.diag).step(
            "(candidate_velocity + " + rows.rhs_of("candidate_normal") + ")", "candidate_normal"
        )
        ring_any = (
            f"RING_ANY(candidate_normal, {M},\n"
            f"    [&](int patch_row) {{ return {rows.parent('patch_row', enclosed=True)}; }},\n"
            f"    [&](int r) {{ return {rows.peek('r')} > 0.0f; }})"
        )
        tangents_zero = " && ".join(f"{rows.peek(f'candidate_normal + {r}')} == 0.0f" for r in range(1, 3))
        if torsion:
            spin = rows.spin_of("candidate_normal", "row_count")
            tangents_zero += f" && ({spin} < 0 || {rows.peek('candidate_normal + 3')} == 0.0f)"
        noop = f"""candidate_noop = old_impulse == 0.0f && new_impulse == 0.0f;
    // Only a unit with tangent rows reads them and its ring's load.
    if (candidate_noop && {owns_tangents("candidate_normal", "        ")}) {{
        candidate_noop = {tangents_zero};
        if (candidate_noop) candidate_noop = !{indent(ring_any, "            ").lstrip()};
    }}"""
        return f"""const int batch_group = lane >> {batch_shift};
const int batch_lane = lane & {batch_lanes - 1};
const int candidate_index = {index} + batch_group;
const bool candidate_valid = candidate_index < {count};
const int candidate_normal = candidate_valid
    ? sparse_contact_serial_normals.data[world * {M} + candidate_index] : -1;
{segment.row_velocity("candidate_velocity", "candidate_normal", "", active="candidate_valid")}
bool candidate_noop = !candidate_valid;
if (candidate_valid) {{
    const float old_impulse = {rows.peek("candidate_normal")};
    float new_impulse = old_impulse;
    if ({rows.diag("candidate_normal")} > 0.0f)
        new_impulse = ProjectionLaw::unilateral(ProjectionLaw::relaxed_step(old_impulse, {step}, omega));
    {noop}
}}
const unsigned noop_lanes = BALLOT(candidate_noop);"""

    row = visit(warp, "            ")
    if not scheduled:
        # The rows hold no schedule: a normal row owning the tangent pair after it heads a contact unit, unless a
        # chain's link marks it (its rows keep their visits, as a scheduled sweep's chain phase visits them).
        head = f"({rows.kind('i')}) == {CONTACT} && {owns_tangents('i', ' ' * 16)}\n{' ' * 16}&& !({independent('i')})"
        serial = f"""        for (int i = 0; i < row_count; ++i) {{
{coupled(head)}
{row}
        }}"""
    elif not speculative_contact_batches:
        serial = f"""        for (int i = 0; i < row_count; ++i) {{
            if (i >= contact_start) {{
                // A tangent row's parent is its unit's head.
                const int unit_head = ({rows.kind("i")}) == {FRICTION} ? {rows.parent("i")} : i;
                if ({independent("unit_head")}) continue;
            }}
{coupled(f"i >= contact_start && ({rows.kind('i')}) == {CONTACT} && {owns_tangents('i', ' ' * 16)}")}
{row}
{inactive_pair(f"i >= contact_start && ({rows.kind('i')}) == {CONTACT} && {owns_tangents('i', ' ' * 16)}", "i")}
        }}"""
    else:
        # Units have one or three rows (four with a spin row): the loop walks real rows and enters each serial unit at
        # its head.
        serial = f"""        const int serial_unit_count = sparse_contact_serial_count.data[world];
        int serial_unit = 0;
        int unit_end = contact_start;
        int tangent_unit = -1;
        for (int i = 0; i < row_count; ++i) {{
            if (i >= unit_end) {{
                int unit_head = -1;
                while (serial_unit < serial_unit_count) {{
                    if ((serial_unit & {batch_contacts - 1}) == 0) {{
{indent(batch("serial_unit", "serial_unit_count"), "                        ")}
                        if (noop_lanes == MASK) {{
                            serial_unit += {batch_contacts};
                            continue;
                        }}
                        serial_unit += FIRST(~noop_lanes) >> {batch_shift};
                    }}
                    unit_head = sparse_contact_serial_normals.data[world * {M} + serial_unit];
                    ++serial_unit;
                    break;
                }}
                if (unit_head < 0) break;
                i = unit_head;
                unit_end = i + 1;
                tangent_unit = -1;
                if ({owns_tangents("i", " " * 20)}) {{
                    unit_end = i + {contact_rows("i")};
                    tangent_unit = i;
                }}
            }}
{coupled("i == tangent_unit")}
{row}
{inactive_pair("i == tangent_unit", "i")}
        }}"""

    declarations, limit_loads, limit_rows, limit_stores = _diagonal_limits(D, P, laws)
    # A diagonal coordinate's factor is sqrt(m): its scale m^-1/2 (zero on the dense articulation's coordinates and
    # where no coordinate is mapped, which keep their velocity).
    diagonal_scale = (
        "const bool dense_coordinate = d >= dense_offset && d < dense_offset + "
        f"{P};\nconst float diagonal_scale = !dense_coordinate && global_dof >= 0\n"
        "    ? sqrtf(diagonal_inverse_mass.data[global_dof]) : 0.0f;"
    )
    sweep_end = end_sweep(solve_policy, iteration="iteration")
    # Where spin rows may appear, every friction triple's metric is formed once per launch, at its first loaded visit
    # (sweep.friction_metric_stage), and its later visits only project.
    metric = ""
    if torsion:
        metric = (
            friction_metric_stage(
                *rows.scan(),
                lambda row: rows.spin_of(row, "row_count"),
                end="row_count",
                record=friction_metric,
                pad="    ",
            )
            + "\n"
            + friction_metric_law("ROW_TRIPLE", friction_metric, "    ")
            + "\n"
        )
    # The phase bound is the raw slot watermark: consumers clamp it to the world's rows.
    contact_start = "min(dense_phase_bounds.data[world * 2 + 1], row_count)" if scheduled else "row_count"
    body = f"""    const int line = dense_first.data[world];
    __builtin_assume(line >= 0);
    const int row_base = line * {M};
    const int lane = LANE;
    int row_count = world_constraint_count.data[world];
    if (row_count > {M}) row_count = {M};
    const int off_dense = row_base;
    const int dof_map_base = world * {D};
    const int dense_offset = dense_offsets.data[world];
    const int dense_group = dense_groups.data[world];
    const int contact_start = {contact_start};

{staging.declarations("    ")}

{declarations}

    for (int i = lane; i < row_count; i += 32) {{
{indent(rows.loads("i"), "        ")}
    }}
    for (int d = lane; d < {D}; d += 32) s_v[d] = 0.0f;
{limit_loads}
    SYNC();
{metric}
    for (int iteration = 0; iteration < iterations; ++iteration) {{
        int iteration_changed = 0;

{limit_rows}
        SYNC();
{independent_phase}
{serial}

{sweep_end}    }}

    for (int i = lane; i < row_count; i += 32) world_impulses.data[off_dense + i] = {rows.peek("i")};
    SYNC();
    // Leave factor coordinates: the dense articulation's change in the walk's coordinates to the step's one leave
    // (tree_factor.walk_leave_kernel, dv = P^T L^-T R^T u); dv = u / sqrt(m) on a diagonal coordinate.
    for (int d = lane; d < {D}; d += 32) {{
        const int global_dof = world_dof_indices.data[dof_map_base + d];
{indent(diagonal_scale, "        ")}
        if (global_dof < 0) continue;
        if (dense_coordinate)
            walk.du.data[global_dof] += s_v[d];
        else
            v_out.data[global_dof] += diagonal_scale > 0.0f ? s_v[d] * diagonal_scale : s_v[d];
    }}
{limit_stores}
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


@cache
def sparse_diagonal_kernel(
    max_constraints: int,
    max_world_dofs: int,
    dense_dofs: int,
    device_arch: str,
    *,
    speculative_contact_batches: bool = False,
    contact_units: bool = False,
    torsion: bool = False,
    solve_policy: SolvePolicy = DEFAULT_POLICY,
    weighted: bool = False,
    material: bool = False,
) -> wp.Kernel:
    """The sparse-diagonal sweep kernel; weighted rows (:func:`.sweep.weight_arguments`) and material contacts
    (:func:`.sweep.material_arguments`) add their operands."""
    snippet, staging = sparse_diagonal_source(
        max_constraints,
        max_world_dofs,
        dense_dofs,
        speculative_contact_batches=speculative_contact_batches,
        contact_units=contact_units,
        torsion=torsion,
        solve_policy=solve_policy,
        weighted=weighted,
        material=material,
    )
    staging.check("the sparse-diagonal sweep", arch_budget(device_arch))
    name = (
        f"sweep_sparse_diagonal_{int(max_constraints)}_{int(max_world_dofs)}_{int(dense_dofs)}"
        + ("_speculative_batches" if speculative_contact_batches else "")
        + ("_units" if contact_units else "")
        + ("_weighted" if weighted else "")
        + ("_material" if material else "")
        + ("_torsion" if torsion else "")
        + solve_policy.suffix
    )
    extra = (
        # Where spin rows may appear, every friction triple's metric (sweep.friction_metric_stage).
        *((RowArgument("friction_metric", wp.array3d[float], after="omega"),) if torsion else ()),
        *weight_arguments(weighted),
        *material_arguments(material),
        *solve_policy.arguments,
    )
    return KernelTemplate(
        name,
        kernel_template.operands(SPARSE_DIAGONAL_ABI, extra),
        snippet,
        kernel_template.items("world"),
        composed=True,
        native_name="sweep_sparse_diagonal_native",
    ).kernel()

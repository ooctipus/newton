# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The sparse-diagonal sweep as a configuration of the sweep emitter (:mod:`.sweep`).

:func:`sparse_diagonal_kernel`: a world's dense group plus diagonal-mass coordinates (:class:`.SparsePairJY`), one
warp per world, its coordinates' fused joint limits, its independent scalar contacts one lane per chain, then its
coupled rows in order (plain, contact triples or contact units, optionally with speculative batches). Its working set
fits the static budget of the device it is built for (:func:`.staging.arch_budget`).
"""

from __future__ import annotations

from functools import cache

import warp as wp

from . import kernel_template
from .device_preludes import RING, cuda_native, lane_group
from .kernel_template import KernelTemplate
from .law_source import law_preludes
from .projection_laws import PROJECTION_LAW
from .response_operator import Lanes, SparsePairJY
from .row_metadata import _validate_dense_metadata_encoding
from .solve_policy import DEFAULT_POLICY, Change, SolvePolicy, bound
from .staging import Buffer, Staging, arch_budget
from .sweep import (
    CONTACT,
    FRICTION,
    JOINT_LIMIT,
    RowContext,
    RowLaws,
    WorldRows,
    change_vote,
    end_sweep,
    friction_gate,
    indent,
    row_visit,
)

# ---------------------------------------------------------------------------------------------------------------------
# Sparse diagonal: a dense group plus two scalar coordinates per row
# ---------------------------------------------------------------------------------------------------------------------

SPARSE_DIAGONAL_ABI = kernel_template.params(
    ("world_constraint_count", wp.array[int]),
    ("world_dof_indices", wp.array2d[int]),
    ("rhs_bias", wp.array2d[float]),
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
    ("dense_J", wp.array3d[float]),
    ("dense_Y", wp.array3d[float]),
    ("sparse_row_dof", wp.array3d[int]),
    ("sparse_row_jy", wp.array3d[float]),
    ("fused_limit_active", wp.array2d[int]),
    ("fused_limit_lower_rhs", wp.array2d[float]),
    ("fused_limit_upper_rhs", wp.array2d[float]),
    ("diagonal_inverse_mass", wp.array[float]),
    ("fused_limit_cfm", float),
    ("iterations", int),
    ("omega", float),
    ("friction_start_iteration", int),
    ("iteration_offset", int),
    ("fused_limit_lower_lambda", wp.array2d[float]),
    ("fused_limit_upper_lambda", wp.array2d[float]),
    ("v_out", wp.array[float]),
)
"""The sparse-diagonal sweep's operands (one world per launch item; a readout follows ``v_out``)."""


def _diagonal_limits(dofs: int, laws: RowLaws) -> tuple[str, str, str, str]:
    """The fused joint limits of a world's diagonal-mass coordinates: declarations, loads, the pass and the stores.

    Coordinate ``d`` has a lower and an upper unilateral row (``J = +1``, ``-1``) on its diagonal response
    ``r = m^-1`` (``A = r + cfm``, the step divides by it). The lane that owns ``d = lane + 32 k`` keeps both
    impulses in registers for the launch; they start at zero and are written back at the end.
    """
    declarations, loads, rows, stores = [], [], [], []
    for k in range((dofs + 31) // 32):
        d = f"lane + {32 * k}" if k else "lane"
        declarations.append(
            f"    int limit_active_{k} = 0;\n"
            f"    float limit_lower_rhs_{k} = 0.0f, limit_upper_rhs_{k} = 0.0f, limit_response_{k} = 0.0f;\n"
            f"    float limit_lower_lambda_{k} = 0.0f, limit_upper_lambda_{k} = 0.0f;"
        )
        loads.append(
            f"    if ({d} < {dofs}) {{\n"
            f"        const int limit_index = world * {dofs} + {d};\n"
            f"        const int global_dof = world_dof_indices.data[dof_map_base + {d}];\n"
            f"        limit_active_{k} = fused_limit_active.data[limit_index];\n"
            f"        limit_lower_rhs_{k} = fused_limit_lower_rhs.data[limit_index];\n"
            f"        limit_upper_rhs_{k} = fused_limit_upper_rhs.data[limit_index];\n"
            f"        if (global_dof >= 0) limit_response_{k} = diagonal_inverse_mass.data[global_dof];\n"
            "    }"
        )
        sides = []
        for bit, side, sign, update in ((1, "lower", "", "+="), (2, "upper", "-", "-=")):
            limit_bound = bound(f"limit_{side}_rhs_{k}", f"fabsf(s_v[{d}])", "old_lambda", diagonal="limit_denom")
            limit = Change("delta_lambda", "limit_bound", str(JOINT_LIMIT), "limit_denom", scaled=True)
            change = change_vote(laws, (limit,), " " * 20)
            sides.append(
                f"            if ((limit_active_{k} & {bit}) != 0) {{\n"
                f"                const float residual = {sign}s_v[{d}] + limit_{side}_rhs_{k};\n"
                f"                const float old_lambda = limit_{side}_lambda_{k};\n"
                f"                const float limit_bound = {limit_bound};\n"
                "                const float new_lambda = ProjectionLaw::unilateral(\n"
                "                    ProjectionLaw::relaxed_step(old_lambda, -residual / limit_denom, omega));\n"
                "                const float delta_lambda = new_lambda - old_lambda;\n"
                f"                limit_{side}_lambda_{k} = new_lambda;\n"
                "                if (delta_lambda != 0.0f) {\n"
                f"{change}\n"
                f"                    s_v[{d}] {update} limit_response_{k} * delta_lambda;\n"
                "                }\n"
                "            }"
            )
        rows.append(
            f"        if ({d} < {dofs} && limit_response_{k} > 0.0f) {{\n"
            f"            const float limit_denom = limit_response_{k} + fused_limit_cfm;\n"
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
    contact_triples: bool = False,
    speculative_contact_batches: bool = False,
    contact_units: bool = False,
    solve_policy: SolvePolicy = DEFAULT_POLICY,
) -> tuple[str, Staging]:
    """The native body of the sparse-diagonal Gauss-Seidel sweep: one warp per world.

    Every sweep runs the world's fused diagonal joint limits (one lane per coordinate), then its independent scalar
    contacts (one lane per chain: a contact whose coordinate no coupled row touches commutes with the coupled rows,
    and its one-product residual rounds as the warp's sum of it does), then the coupled rows in row order with
    :class:`.SparsePairJY`'s warp layout. ``contact_triples`` schedules point contacts (a normal and two tangent
    rows each), ``contact_units`` patch contacts (a normal, and its tangent pair only when it owns an anchor); both
    skip an independent contact in the serial order and a pair whose patch carries no load. With
    ``speculative_contact_batches`` the coupled contacts come from the world's serial list, and a batch of them
    (one per segment of the warp) is tested first: a batch whose normals stay zero without load is skipped whole,
    otherwise the sweep enters the batch at its first active contact.

    Returns the body and its staging.
    """
    M, D, P = int(max_constraints), int(max_world_dofs), int(dense_dofs)
    if contact_triples and contact_units:
        raise ValueError("contact triples and contact units are exclusive schedules")
    if speculative_contact_batches and (not (contact_triples or contact_units) or P + 2 > 16):
        raise ValueError(
            "speculative contact batches require contact triples or units with at most 16 response entries"
        )
    _validate_dense_metadata_encoding(M)
    batch_lanes = 8 if P + 2 <= 8 else 16
    batch_shift = batch_lanes.bit_length() - 1
    batch_contacts = 32 // batch_lanes
    serial_stride = M if contact_units else (M + 2) // 3
    scheduled = contact_triples or contact_units

    warp = Lanes(32)
    laws = RowLaws(drive=False, regularize=False, policy=solve_policy)
    rows = WorldRows(M, lanes=warp)
    staging = Staging((Buffer("s_v", "float", D), *rows.buffers()), threads=32)

    def visit(lanes: Lanes, pad: str, *, kind: int | None = None, parent: str | None = None) -> str:
        """Row ``i``'s visit in a layout: the warp's (every lane stores, the warp synchronizes after the apply) or one
        thread's (an independent scalar contact on ``scalar_coord``). A ``kind`` (and a friction row's ``parent``)
        known where the visit is emitted replaces the row's metadata, so the visit compiles to that kind's law; a
        known kind other than friction admits no pair."""
        diagonal = lambda row: "denom" if row == "i" else rows.diag(row)  # noqa: E731
        op = SparsePairJY(P, M, lanes, diagonal=diagonal)
        ctx = RowContext(
            row="i",
            kind="row_type",
            operator=op,
            load=rows.peek,
            store=lambda row, value: f"{rows.peek(row)} = {value};",
            rhs=rows.rhs_of,
            mu=rows.mu,
            weight=rows.weight,
            parent="parent",
            ring_load=rows.ring_load,
            lanes=lanes,
            inline_velocity_limit=False,
            # Every lane (or the one thread) stores what it computed; a pair finishes its second tangent in its branch,
            # on the entries it kept (SparsePairJY.pair_entries).
            owned=False,
            friction=kind in (None, FRICTION),
        )
        return f"{pad}const int row_type = {rows.kind('i') if kind is None else kind};\n" + row_visit(
            ctx,
            laws,
            parent=rows.parent("i") if parent is None else parent,
            check="denom",
            declare=f"const float denom = {rows.diag('i')};",
            entries=op.consume("i", ""),
            pad=pad,
        )

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
        """Skip a contact's tangent rows when they hold nothing and its patch carries no load."""
        return (
            f"            if ({condition}\n"
            f"                && {rows.peek('i + 1')} == 0.0f\n"
            f"                && {rows.peek('i + 2')} == 0.0f) {{\n"
            f"                const float patch_load = {indent(rows.ring_load('i'), '                ').lstrip()};\n"
            f"                if (patch_load <= 0.0f) {advance} += 2;\n"
            "            }"
        )

    # Independent scalar chains: one lane per chain, its contacts in chain order. Schedule construction excludes every
    # coordinate a coupled row touches, so they commute with the coupled rows. A contact's normal links the chain's
    # next contact through its free slot (-2 ends the chain). A contact is a normal row and, for a triple or a unit
    # that owns an anchor, the tangent pair after it, which names the normal as parent: the visits' kinds are known
    # here, so each compiles to its own law. The pair is solved at its first tangent; the second tangent is only
    # zeroed while friction is delayed. A visit that skips its row leaves its block (do-while), not the chain.
    independent_phase = ""
    if scheduled:
        pad = " " * 16
        tangents = f"""{pad}do {{
{pad}    const int i = head + 1;
{visit(Lanes(1), pad + "    ", kind=FRICTION, parent="head")}
{pad}}} while (false);
{friction_gate(str(FRICTION), f"{rows.peek('head + 2')} = 0.0f;", "", pad, skip=False)}"""
        if contact_units:
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
        segment = SparsePairJY(P, M, Lanes(batch_lanes, "batch_lane", segment=True), diagonal=rows.diag)
        step = segment.step("(candidate_velocity + " + rows.rhs_of("candidate_normal") + ")", "candidate_normal")
        ring_any = (
            f"RING_ANY(candidate_normal, {M},\n"
            f"    [&](int patch_row) {{ return {rows.parent('patch_row', enclosed=True)}; }},\n"
            f"    [&](int r) {{ return {rows.peek('r')} > 0.0f; }})"
        )
        tangents_zero = f"{rows.peek('candidate_normal + 1')} == 0.0f && {rows.peek('candidate_normal + 2')} == 0.0f"
        if contact_units:
            noop = f"""candidate_noop = old_impulse == 0.0f && new_impulse == 0.0f;
    // Only a unit with tangent rows reads them and its ring's load.
    if (candidate_noop && {owns_tangents("candidate_normal", "        ")}) {{
        candidate_noop = {tangents_zero};
        if (candidate_noop) candidate_noop = !{indent(ring_any, "            ").lstrip()};
    }}"""
        else:
            noop = f"""candidate_noop = old_impulse == 0.0f && new_impulse == 0.0f && {tangents_zero};
    if (candidate_noop) candidate_noop = !{indent(ring_any, "        ").lstrip()};"""
        return f"""const int batch_group = lane >> {batch_shift};
const int batch_lane = lane & {batch_lanes - 1};
const int candidate_index = {index} + batch_group;
const bool candidate_valid = candidate_index < {count};
const int candidate_normal = candidate_valid
    ? sparse_contact_serial_normals.data[world * {serial_stride} + candidate_index] : -1;
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
        serial = f"""        for (int i = 0; i < row_count; ++i) {{
{row}
        }}"""
    elif contact_triples and not speculative_contact_batches:
        serial = f"""        for (int i = 0; i < row_count; ++i) {{
            if (i >= contact_start) {{
                const int normal = contact_start + ((i - contact_start) / 3) * 3;
                if ({independent("normal")}) {{
                    i = normal + 2;
                    continue;
                }}
            }}
{row}
{inactive_pair("i >= contact_start && (i - contact_start) % 3 == 0", "i")}
        }}"""
    elif contact_triples:
        serial = f"""        const int serial_contact_count = sparse_contact_serial_count.data[world];
        const int serial_row_end = contact_start + serial_contact_count * 3;
        for (int serial_row = 0; serial_row < serial_row_end; ++serial_row) {{
            int i = serial_row;
            int contact_component = -1;
            if (serial_row >= contact_start) {{
                const int contact_index = (serial_row - contact_start) / 3;
                if (serial_row - contact_start == contact_index * 3 && (contact_index & {batch_contacts - 1}) == 0) {{
{indent(batch("contact_index", "serial_contact_count"), "                    ")}
                    if (noop_lanes == MASK) {{
                        const int remaining_contacts = serial_contact_count - contact_index;
                        serial_row += (remaining_contacts < {batch_contacts} ? remaining_contacts : {batch_contacts}) * 3 - 1;
                        continue;
                    }}
                    serial_row += (FIRST(~noop_lanes) >> {batch_shift}) * 3;
                }}
                contact_component = (serial_row - contact_start) % 3;
                i = sparse_contact_serial_normals.data[world * {serial_stride} + (serial_row - contact_start) / 3]
                    + contact_component;
            }}
{row}
{inactive_pair("contact_component == 0", "serial_row")}
        }}"""
    elif not speculative_contact_batches:
        serial = f"""        for (int i = 0; i < row_count; ++i) {{
            if (i >= contact_start) {{
                // A tangent row's parent is its unit's head.
                const int unit_head = ({rows.kind("i")}) == {FRICTION} ? {rows.parent("i")} : i;
                if ({independent("unit_head")}) continue;
            }}
{row}
{inactive_pair(f"i >= contact_start && ({rows.kind('i')}) == {CONTACT} && {owns_tangents('i', ' ' * 16)}", "i")}
        }}"""
    else:
        # Units have one or three rows: the loop walks real rows and enters each serial unit at its head.
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
                    unit_head = sparse_contact_serial_normals.data[world * {serial_stride} + serial_unit];
                    ++serial_unit;
                    break;
                }}
                if (unit_head < 0) break;
                i = unit_head;
                unit_end = i + 1;
                tangent_unit = -1;
                if ({owns_tangents("i", " " * 20)}) {{
                    unit_end = i + 3;
                    tangent_unit = i;
                }}
            }}
{row}
{inactive_pair("i == tangent_unit", "i")}
        }}"""

    declarations, limit_loads, limit_rows, limit_stores = _diagonal_limits(D, laws)
    sweep_end = end_sweep(solve_policy, iteration="iteration")
    # The phase bound is the raw slot watermark: consumers clamp it to the world's rows.
    contact_start = "min(dense_phase_bounds.data[world * 2 + 1], row_count)" if scheduled else "row_count"
    body = f"""    const int lane = LANE;
    int row_count = world_constraint_count.data[world];
    if (row_count > {M}) row_count = {M};
    const int row_base = world * {M};
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
    for (int d = lane; d < {D}; d += 32) {{
        const int global_dof = world_dof_indices.data[dof_map_base + d];
        s_v[d] = global_dof >= 0 ? v_out.data[global_dof] : 0.0f;
    }}
{limit_loads}
    SYNC();

    for (int iteration = 0; iteration < iterations; ++iteration) {{
        const int global_iter = iteration_offset + iteration;
        int iteration_changed = 0;

{limit_rows}
        SYNC();
{independent_phase}
{serial}

{sweep_end}    }}

    for (int i = lane; i < row_count; i += 32) world_impulses.data[off_dense + i] = {rows.peek("i")};
    for (int d = lane; d < {D}; d += 32) {{
        const int global_dof = world_dof_indices.data[dof_map_base + d];
        if (global_dof >= 0) v_out.data[global_dof] = s_v[d];
    }}
{limit_stores}
#endif
"""
    preludes = (lane_group(32), RING, *law_preludes(PROJECTION_LAW), *solve_policy.preludes)
    return cuda_native(body, preludes=preludes), staging


@cache
def sparse_diagonal_kernel(
    max_constraints: int,
    max_world_dofs: int,
    dense_dofs: int,
    device_arch: str,
    *,
    contact_triples: bool = False,
    speculative_contact_batches: bool = False,
    contact_units: bool = False,
    solve_policy: SolvePolicy = DEFAULT_POLICY,
) -> wp.Kernel:
    """The sparse-diagonal sweep kernel."""
    snippet, staging = sparse_diagonal_source(
        max_constraints,
        max_world_dofs,
        dense_dofs,
        contact_triples=contact_triples,
        speculative_contact_batches=speculative_contact_batches,
        contact_units=contact_units,
        solve_policy=solve_policy,
    )
    staging.check("the sparse-diagonal sweep", arch_budget(device_arch))
    name = (
        f"sweep_sparse_diagonal_{int(max_constraints)}_{int(max_world_dofs)}_{int(dense_dofs)}"
        + ("_contact_groups" if contact_triples else "")
        + ("_speculative_batches" if speculative_contact_batches else "")
        + ("_units" if contact_units else "")
        + solve_policy.suffix
    )
    return KernelTemplate(
        name,
        kernel_template.operands(SPARSE_DIAGONAL_ABI, solve_policy.arguments),
        snippet,
        kernel_template.WORLDS,
        composed=True,
        native_name="sweep_sparse_diagonal_native",
    ).kernel()

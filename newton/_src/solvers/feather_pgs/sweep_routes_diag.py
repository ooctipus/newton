# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The sparse-diagonal, tiled-row and split free-body sweeps as configurations of the sweep emitter (:mod:`.sweep`).

- :func:`sparse_diagonal_kernel`: a world's dense group plus diagonal-mass coordinates (:class:`.SparsePairJY`), one
  warp per world, its coordinates' fused joint limits, its independent scalar contacts one lane per chain, then its
  coupled rows in order (plain, contact triples or contact units, optionally with speculative batches);
- :func:`tiled_row_kernel`: split mode's assembled Delassus rows (:class:`.Assembled`), one warp per world, in
  impulse space;
- :func:`mf_split_kernel`: split mode's free-body rows (:class:`.FreeBodyZ` with one thread per row), their bodies'
  velocities staged through the world's local body map.

Every kernel's working set fits the static budget of the device it is built for (:func:`.staging.arch_budget`).
"""

from __future__ import annotations

from functools import cache

import warp as wp

from . import kernel_template
from .device_preludes import RING, cuda_native, lane_group
from .kernel_template import KernelTemplate
from .kernels import BodyFactors
from .law_source import law_preludes
from .projection_laws import PROJECTION_LAW
from .response_operator import Assembled, FreeBodyZ, Lanes, SparsePairJY, Velocity
from .row_metadata import _validate_dense_metadata_encoding
from .solve_policy import EXACT_POLICY, SolvePolicy, sweep_exit
from .staging import Buffer, Staging, arch_budget
from .sweep import (
    CONTACT,
    FRICTION,
    JOINT_LIMIT,
    VELOCITY_LIMIT,
    RowContext,
    RowLaws,
    WorldRows,
    change_vote,
    end_sweep,
    friction_gate,
    indent,
    row_visit,
    speed_row,
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
            change = change_vote(laws, (("delta_lambda", "limit_denom", str(JOINT_LIMIT)),), " " * 20)
            sides.append(
                f"            if ((limit_active_{k} & {bit}) != 0) {{\n"
                f"                const float residual = {sign}s_v[{d}] + limit_{side}_rhs_{k};\n"
                f"                const float old_lambda = limit_{side}_lambda_{k};\n"
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
    solve_policy: SolvePolicy = EXACT_POLICY,
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
    sweep_end = end_sweep(
        solve_policy,
        iteration="iteration",
        exact="""        if (global_iter >= friction_start_iteration
            && BALLOT(iteration_changed != 0) == 0u) break;
""",
    )
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
    solve_policy: SolvePolicy = EXACT_POLICY,
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


# ---------------------------------------------------------------------------------------------------------------------
# Tiled row: split mode's assembled Delassus rows, in impulse space
# ---------------------------------------------------------------------------------------------------------------------

TILED_ROW_ABI = kernel_template.params(
    ("world_constraint_count", wp.array[int]),
    ("world_diag", wp.array2d[float]),
    ("world_C", wp.array3d[float]),
    ("world_rhs", wp.array2d[float]),
    ("world_impulses", wp.array2d[float]),
    ("iterations", int),
    ("omega", float),
    ("world_row_type", wp.array2d[int]),
    ("world_row_parent", wp.array2d[int]),
    ("world_row_mu", wp.array2d[float]),
    ("world_row_w", wp.array2d[float]),
    ("regularize", int),
    ("friction_start_iteration", int),
    ("iteration_offset", int),
)
"""The tiled-row sweep's operands (one world per launch item)."""


def tiled_row_source(
    max_constraints: int,
    *,
    row_budget: int | None = None,
    min_rows: int = 0,
    staging: str = "shared",
    solve_policy: SolvePolicy = EXACT_POLICY,
) -> tuple[str, Staging]:
    """The native body of split mode's Gauss-Seidel sweep over a world's assembled Delassus rows: one warp per world.

    The rows' impulses, right-hand sides, diagonals, types, parents and friction coefficients are staged, and ``C``
    as ``staging`` names (:class:`.Assembled`); every lane runs every row's law on its own copy of the impulses.
    Contact, position-limit and velocity-limit rows are unilateral, a friction pair is solved at its first tangent on
    its patch ring's pooled load, every other row is bilateral; a contact's step takes its regularization weight
    when ``regularize`` is set. A world stops under the solve policy (:func:`.sweep.end_sweep`).

    A row budget (C64 or C128 only, 16, 32 or the capacity rows) stages that many rows of the capacity's strides; a
    small budget owns the worlds of ``min_rows < m <= row_budget`` rows whose friction metadata stays in their rows,
    the capacity budget the larger worlds and the small worlds the small budget declines.

    Returns the body and its staging.
    """
    if staging not in Assembled.STAGINGS:
        raise ValueError(f"staging must be one of {Assembled.STAGINGS}, got {staging!r}")
    if row_budget is not None:
        if (
            max_constraints not in (64, 128)
            or row_budget not in (16, 32, max_constraints)
            or not 0 <= min_rows < row_budget
            or staging != "shared"
        ):
            raise ValueError(
                "Dense row budgets require C64/C128, budget 16/32/capacity, 0 <= min_rows < budget and shared staging"
            )
    elif min_rows != 0:
        raise ValueError("min_rows requires an explicit row_budget")
    M = int(max_constraints)
    rows = M if row_budget is None else int(row_budget)
    small_budget = row_budget is not None and row_budget < M
    op = Assembled(rows, str(M), staging, Lanes(32), diagonal=lambda row: "denom" if row == "i" else f"s_diag[{row}]")
    triangle = (Buffer("s_Ctri", "float", op.triangle),) if op.shared else ()
    stage = Staging(
        (
            *triangle,
            *(Buffer(name, "float", rows) for name in ("s_lam", "s_rhs", "s_diag")),
            Buffer("s_rtype", "int", rows),
            Buffer("s_parent", "int", rows),
            Buffer("s_mu", "float", rows),
        ),
        threads=32,
    )
    laws = RowLaws(drive=False, regularize=True, policy=solve_policy)
    ctx = RowContext(
        row="i",
        kind="row_type",
        operator=op,
        load=lambda row: f"s_lam[{row}]",
        store=lambda row, value: f"s_lam[{row}] = {value};",
        rhs=lambda row: f"s_rhs[{row}]",
        mu=lambda row: f"s_mu[{row}]",
        weight=lambda row: f"world_row_w.data[off1 + {row}]",
        parent="parent",
        ring_load=lambda parent: (
            f"RING_LOAD(s_lam[{parent}], {parent}, {rows},\n"
            "    [&](int r) { return s_parent[r]; }, [&](int r) { return s_lam[r]; })"
        ),
        lanes=Lanes(32),
        contact_kinds=(CONTACT, JOINT_LIMIT, VELOCITY_LIMIT),
        owned=False,
    )
    admission = ""
    if row_budget is not None:
        if small_budget:
            admission += f"    if (m <= {min_rows} || m > {row_budget}) return;\n"
        admission += f"""    bool unsupported = false;
    if (m <= 32) {{
        if (lane < m) {{
            const int p = world_row_parent.data[world * {M} + lane];
            // Normal rows can link additional patch normals. Keep every link
            // inside the staged prefix, including links reached transitively.
            unsupported = p >= m;
            if (world_row_type.data[world * {M} + lane] == {FRICTION}) {{
                // Friction still projects when its diagonal is nonpositive.
                // Bound p before adding to avoid integer overflow.
                unsupported = p < 0 || p > m - 3 || (lane != p + 1 && lane != p + 2);
            }}
        }}
        unsupported = BALLOT(unsupported) != 0u;
    }}
"""
        admission += (
            "    if (unsupported) return;\n" if small_budget else f"    if (m <= {min_rows} && !unsupported) return;\n"
        )
    # A small budget stages the world's rows, one per lane; otherwise every row of the budget (friction metadata may
    # read past m).
    staged = "if (lane < m) {" if small_budget else f"for (int r = lane; r < {rows}; r += 32) {{"
    r = "lane" if small_budget else "r"
    row = row_visit(
        ctx, laws, parent="s_parent[i]", check="denom", declare="const float denom = s_diag[i];", pad="            "
    )
    body = f"""    const int lane = LANE;

    const int m = world_constraint_count.data[world];
    if (m == 0) return;
{admission}
{stage.declarations("    ")}
    const int off1 = world * {M};
    const int off2 = world * {M * M};

    {staged}
        s_lam[{r}] = world_impulses.data[off1 + {r}];
        s_rhs[{r}] = world_rhs.data[off1 + {r}];
        s_diag[{r}] = world_diag.data[off1 + {r}];
        s_rtype[{r}] = world_row_type.data[off1 + {r}];
        s_parent[{r}] = world_row_parent.data[off1 + {r}];
        s_mu[{r}] = world_row_mu.data[off1 + {r}];
    }}
{op.stage("    ")}
    SYNC();

{op.prepare("    ")}

    for (int iter = 0; iter < iterations; iter++) {{
        const int global_iter = iteration_offset + iter;
        int iteration_changed = 0;
        for (int i = 0; i < m; i++) {{
            const int row_type = s_rtype[i];
{row}
        }}
{end_sweep(solve_policy, iteration="iter")}    }}

    {staged}
        world_impulses.data[off1 + {r}] = s_lam[{r}];
    }}
#endif
"""
    preludes = (lane_group(32), RING, *law_preludes(PROJECTION_LAW), *solve_policy.preludes)
    return cuda_native(body, preludes=preludes), stage


def tiled_row_staging(max_constraints: int, limit: int) -> tuple[str, str]:
    """Choose where a tiled-row sweep of ``max_constraints`` rows reads the Delassus operator under a static budget.

    The packed triangle stays in shared memory while the declared staging (:func:`tiled_row_source`) fits ``limit``,
    the device's static shared memory per block [B] (:func:`.staging.arch_budget`); past it, the warp reads each
    Delassus row from global memory and stages only the row vectors.

    Returns:
        The staging, one of :attr:`.Assembled.STAGINGS`, and the reason.

    Raises:
        ValueError: If even the row vectors exceed the budget.
    """
    shared = tiled_row_source(max_constraints, staging="shared")[1].bytes
    if shared <= limit:
        return "shared", (
            f"The packed Delassus triangle of {max_constraints} rows and the row vectors take {shared} of the "
            f"device's {limit} bytes of static shared memory per block."
        )
    rows = tiled_row_source(max_constraints, staging="global")[1].bytes
    if rows <= limit:
        return "global", (
            f"The packed Delassus triangle of {max_constraints} rows would take {shared} bytes of static shared "
            f"memory per block, more than the device's {limit}: the sweep reads Delassus rows from global memory and "
            f"stages only the row vectors ({rows} bytes)."
        )
    raise ValueError(
        f"dense_max_constraints={max_constraints} needs {rows} bytes of shared memory per block for the tiled_row "
        f"sweep's row vectors, more than the device's {limit}; use pgs_kernel='loop' or fewer dense rows"
    )


@cache
def tiled_row_kernel(
    max_constraints: int,
    device_arch: str,
    *,
    row_budget: int | None = None,
    min_rows: int = 0,
    staging: str = "shared",
    solve_policy: SolvePolicy = EXACT_POLICY,
) -> wp.Kernel:
    """The tiled-row sweep kernel; a readout's operand follows the rows' operands.

    Raises:
        ValueError: If the staging's static shared memory exceeds the device's budget
            (:func:`.staging.arch_budget`), which the solver's staging choice never asks for.
    """
    snippet, declared = tiled_row_source(
        max_constraints, row_budget=row_budget, min_rows=min_rows, staging=staging, solve_policy=solve_policy
    )
    declared.check(f"the tiled-row sweep of {max_constraints} rows with {staging} staging", arch_budget(device_arch))
    suffix = "" if row_budget is None else f"_b{row_budget}_m{min_rows}"
    if staging == "global":
        suffix += "_global"
    return KernelTemplate(
        f"sweep_tiled_row_{max_constraints}{suffix}{solve_policy.suffix}",
        kernel_template.operands(TILED_ROW_ABI, solve_policy.arguments),
        snippet,
        dispatch=kernel_template.WORLDS,
        composed=True,
        native_name="sweep_tiled_row_native",
    ).kernel()


# ---------------------------------------------------------------------------------------------------------------------
# Split free bodies: one thread solves a world's free-body rows, velocities staged through its local body map
# ---------------------------------------------------------------------------------------------------------------------

MF_SPLIT_ABI = kernel_template.params(
    ("mf_constraint_count", wp.array[int]),
    ("mf_body_count", wp.array[int]),
    ("mf_body_dof_start", wp.array2d[int]),
    ("mf_local_body_a", wp.array2d[int]),
    ("mf_local_body_b", wp.array2d[int]),
    ("mf_Z_a", wp.array3d[float]),
    ("mf_Z_b", wp.array3d[float]),
    ("factors", BodyFactors),
    ("mf_eff_mass_inv", wp.array2d[float]),
    ("mf_rhs", wp.array2d[float]),
    ("mf_row_type", wp.array2d[int]),
    ("mf_row_parent", wp.array2d[int]),
    ("mf_row_mu", wp.array2d[float]),
    ("mf_row_w", wp.array2d[float]),
    ("mf_impulses", wp.array2d[float]),
    ("v_out", wp.array[float]),
    ("iterations", int),
    ("omega", float),
    ("regularize", int),
    ("friction_start_iteration", int),
    ("iteration_offset", int),
)
"""The split free-body sweep's operands (one world per launch item)."""


def mf_split_source(
    mf_max_constraints: int,
    max_mf_bodies: int,
    *,
    has_velocity_limit_rows: bool = False,
    solve_policy: SolvePolicy = EXACT_POLICY,
) -> tuple[str, Staging]:
    """The native body of split mode's free-body Gauss-Seidel sweep: one warp per world, its first lane solving.

    The warp stages the world's bodies' velocities (six coordinates per body of the local body map,
    ``mf_body_dof_start``) in factor coordinates (:meth:`.FreeBodyZ.enter`, back with :meth:`.FreeBodyZ.leave` before
    the write) and the rows' impulses; lane 0 runs every sweep over the rows in order with :class:`.FreeBodyZ`'s
    one-thread layout, each row's bodies at their local slots (``mf_local_body_a``/``_b``).
    Contact and limit rows are unilateral, a friction pair is solved at its first tangent, every other row is
    bilateral. With ``has_velocity_limit_rows``, a velocity-limit row is its body's speed row
    (:func:`.sweep.speed_row`, stateless), in row order: the rows occupy the tail of the world's rows. A world stops
    under the solve policy, lane 0 voting alone.

    Returns the body and its staging.
    """
    M, B = int(mf_max_constraints), int(max_mf_bodies)
    serial = Lanes(1)
    laws = RowLaws(drive=False, regularize=True, policy=solve_policy)
    op = FreeBodyZ(
        serial,
        velocity=Velocity("s_vel"),
        inverse=lambda row: "mf_diag" if row == "i" else f"mf_eff_mass_inv.data[c_off + {row}]",
        factor=lambda first: f"s_factor[({first}) / 6]",
    )
    stage = Staging(
        (
            Buffer("s_vel", "float", B * 6),
            Buffer("s_impulse", "float", M),
            Buffer("s_dof_start", "int", B),
            Buffer("s_factor", "int", B),
        ),
        threads=32,
    )
    ctx = RowContext(
        row="i",
        kind="mf_rt",
        operator=op,
        load=lambda row: f"s_impulse[{row}]",
        store=lambda row, value: f"s_impulse[{row}] = {value};",
        rhs=lambda row: f"mf_rhs.data[c_off + {row}]",
        mu=lambda row: f"mf_row_mu.data[c_off + {row}]",
        weight=lambda row: f"mf_row_w.data[c_off + {row}]",
        parent="parent",
        ring_load=lambda parent: (
            f"RING_LOAD(s_impulse[{parent}], {parent}, {M},\n"
            "    [&](int r) { return mf_row_parent.data[c_off + r]; }, [&](int r) { return s_impulse[r]; })"
        ),
        lanes=serial,
        contact_kinds=(CONTACT, JOINT_LIMIT),
    )
    pad = "                "
    speed_rows = ""
    if has_velocity_limit_rows:
        speed_rows = f"""{pad}if (mf_rt == {VELOCITY_LIMIT}) {{
{pad}    // The body's linear (J = e_0) or angular (J = e_3) speed, bounded along the velocity.
{pad}    const float mf_diag = mf_eff_mass_inv.data[c_off + i];
{pad}    const int dof_a = mf_local_body_a.data[c_off + i] * 6;
{pad}    if (mf_diag <= 0.0f || dof_a < 0) continue;
{pad}    const int speed_axis = mf_Z_a.data[mf6_base + i * 6 + 3] != 0.0f ? 3 : 0;
{speed_row(op, laws, load="s_impulse[i]", store="s_impulse[i] = speed_row[0];", lanes=serial, rhs="mf_rhs.data[c_off + i]", inverse="mf_diag", pad=pad + "    ")}
{pad}    continue;
{pad}}}"""
    row = row_visit(
        ctx,
        laws,
        parent="mf_row_parent.data[c_off + i]",
        check="mf_diag",
        declare="const float mf_diag = mf_eff_mass_inv.data[c_off + i];",
        entries=(
            "const int dof_a = mf_local_body_a.data[c_off + i] * 6;\n"
            "const int dof_b = mf_local_body_b.data[c_off + i] * 6;"
        ),
        pad=pad,
    )
    # Lane 0 sweeps alone: its own vote is the world's.
    quiet = "iteration_changed == 0"
    sweep_end = sweep_exit(
        solve_policy,
        f"""            if (global_iter >= friction_start_iteration && {quiet}) break;
""",
        quiet=quiet,
        gate="global_iter >= friction_start_iteration",
        iteration="iter",
        argmax=None,
        indent="            ",
    )
    body = f"""    const int lane = LANE;

    int m = mf_constraint_count.data[world];
    if (m == 0) return;
    if (m > {M}) m = {M};
    int n_bodies = mf_body_count.data[world];
    if (n_bodies > {B}) n_bodies = {B};

{stage.declarations("    ")}
    const int body_off = world * {B};
    const int c_off = world * {M};
    const int mf6_base = c_off * 6;

    for (int b = lane; b < n_bodies; b += 32) {{
        const int dof = mf_body_dof_start.data[body_off + b];
        const int factor = factors.dof_factor.data[dof];
        s_dof_start[b] = dof;
        s_factor[b] = factor;
        for (int k = 0; k < 6; k++) s_vel[b * 6 + k] = v_out.data[dof + k];
{indent(op.enter(lambda k: f"s_vel[b * 6 + {k}]", "factor"), "        ")}
    }}
    for (int i = lane; i < m; i += 32) s_impulse[i] = mf_impulses.data[c_off + i];
    SYNC();

    if (lane == 0) {{
        for (int iter = 0; iter < iterations; iter++) {{
            const int global_iter = iteration_offset + iter;
            int iteration_changed = 0;
            for (int i = 0; i < m; i++) {{
                const int mf_rt = mf_row_type.data[c_off + i];
{speed_rows}
{row}
            }}
{sweep_end}        }}
    }}
    SYNC();

    for (int b = lane; b < n_bodies; b += 32) {{
        const int dof = s_dof_start[b];
{indent(op.leave(lambda k: f"s_vel[b * 6 + {k}]", "s_factor[b]"), "        ")}
        for (int k = 0; k < 6; k++) v_out.data[dof + k] = s_vel[b * 6 + k];
    }}
    for (int i = lane; i < m; i += 32) mf_impulses.data[c_off + i] = s_impulse[i];
#endif
"""
    preludes = (lane_group(32), RING, *law_preludes(PROJECTION_LAW), *solve_policy.preludes)
    return cuda_native(body, preludes=preludes), stage


@cache
def mf_split_kernel(
    mf_max_constraints: int,
    max_mf_bodies: int,
    device_arch: str,
    *,
    has_velocity_limit_rows: bool = False,
    solve_policy: SolvePolicy = EXACT_POLICY,
) -> wp.Kernel:
    """The split free-body sweep kernel."""
    snippet, staging = mf_split_source(
        mf_max_constraints, max_mf_bodies, has_velocity_limit_rows=has_velocity_limit_rows, solve_policy=solve_policy
    )
    staging.check("the split free-body sweep", arch_budget(device_arch))
    name = (
        f"sweep_mf_split_{mf_max_constraints}_{max_mf_bodies}"
        + ("_vlim" if has_velocity_limit_rows else "")
        + solve_policy.suffix
    )
    return KernelTemplate(
        name,
        kernel_template.operands(MF_SPLIT_ABI, solve_policy.arguments),
        snippet,
        dispatch=kernel_template.WORLDS,
        composed=True,
        native_name="sweep_mf_split_native",
    ).kernel()

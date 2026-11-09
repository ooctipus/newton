# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The sparse-diagonal sweep as a configuration of the sweep emitter (:mod:`.sweep`).

:class:`DiagonalSweep`, built by the one factory (:func:`.sweep.sweep_kernel`): a world's dense group plus diagonal-mass
coordinates as whitened rows
(:func:`slot_rows`, a :class:`.SparseFactor`), one warp per world, on the velocity change from the predictor in factor coordinates, from zero:
``du = R^-T L^T P dv`` on the dense articulation (the walk's factor and its slips' release, the frame its rows were whitened
in) and ``du = sqrt(m) dv`` on a diagonal coordinate, left once and added to the predictor. Each sweep runs its coordinates' fused joint limits, its independent scalar contacts a level at a time, a lane a contact, then its
coupled rows in order (plain or contact units, optionally with speculative batches), a loaded coupled
contact unit by the unit law (:func:`.sweep.contact_unit`). Its working set fits the static
budget of the device it is built for (:func:`.memory.arch_budget`).
"""

from __future__ import annotations

import dataclasses
from typing import ClassVar

import warp as wp

from ...core.execution import Work, device_lanes
from ...core.memory import Buffer, Staging
from . import kernel_template
from .factor.walk import WalkTrees
from .kernel_template import Dispatch
from .response_operator import Lanes, ResponseOperator, SparseFactor, Velocity
from .row_inputs import RowArgument
from .row_laws import ROW_CONTACT, ROW_FRICTION, ROW_JOINT_LIMIT, ROW_JOINT_VELOCITY_LIMIT
from .sweep import (
    EXACT,
    ITERATION,
    SEQUENTIAL,
    Diagonal,
    Levels,
    RowContext,
    RowLaws,
    SlotOrder,
    SweepConfig,
    UnitZ,
    WorldRows,
    contact_unit,
    end_sweep,
    friction_metric,
    friction_metric_law,
    friction_metric_stage,
    impulse_seed,
    indent,
    level_loop,
    row_step,
    row_update,
    velocity_limit_pass,
)

# ---------------------------------------------------------------------------------------------------------------------
# Sparse diagonal: a dense group plus two scalar coordinates per row
# ---------------------------------------------------------------------------------------------------------------------

SPARSE_DIAGONAL_ABI = kernel_template.params(
    ("world", int),
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
    ("sparse_contact_level_count", wp.array[int]),
    ("sparse_contact_level_offsets", wp.array2d[int]),
    ("sparse_contact_level_units", wp.array2d[int]),
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
"""The sparse-diagonal sweep's native ABI (one world per launch item; a readout follows ``v_out``)."""


@dataclasses.dataclass(frozen=True)
class LimitZ(ResponseOperator):
    """A fused joint-limit row's response on a diagonal coordinate (``coordinate``, of factor ``sqrt(m)``): an
    unstored whitened row, ``Z = s`` for the lower row (``J = +1``) and ``Z = -s`` for the upper (``J = -1``), with
    ``s = m^-1/2`` (``scale``); one thread's (the lane that owns the coordinate).

    ``dot`` reduces the coordinate's scaled velocity ``s u``, and a row's velocity term is it with the row's sign
    (:meth:`term`); ``axpy`` adds or subtracts ``s delta``. These are ``Z = -s``'s products negated exactly, in the
    form whose machine code is the sweep's."""

    coordinate: str
    scale: str
    upper: bool = False

    def term(self, out: str) -> str:
        """The row's velocity term ``Z . u`` from ``dot``'s ``s u`` in ``out``."""
        return f"-{out}" if self.upper else out

    def dot(self, sums, rows: dict[str, str], indent: str, **_) -> str:
        return "\n".join(f"{indent}const float {out} = {self.scale} * s_v[{self.coordinate}];" for out, _, _ in sums)

    def axpy(self, changes, indent: str, **_) -> str:
        update = "-=" if self.upper else "+="
        return "".join(f"\n{indent}s_v[{self.coordinate}] {update} {self.scale} * {delta};" for _, delta in changes)


@dataclasses.dataclass(frozen=True)
class LimitRows:
    """The fused joint-limit rows of the coordinates a lane owns, ``d = lane + L k`` for ``L`` lanes: each coordinate's ``lower`` and
    ``upper`` row, their impulses and right-hand sides in the lane's registers for the launch
    (``limit_{row}_lambda_{k}``, ``limit_{row}_rhs_{k}``: the rows' incident velocities ``+-v_hat``,
    :func:`.row_kernels.prepare_fused_diagonal_joint_limits`), their diagonal ``A = s^2 + cfm`` (``limit_denom``), which
    the step divides by. Each lane stores its own rows."""

    k: int
    lanes: ClassVar[Lanes] = Lanes(1)

    def load(self, row: str) -> str:
        return f"limit_{row}_lambda_{self.k}"

    def store(self, row: str, value: str) -> str:
        return f"{self.load(row)} = {value};"

    def rhs_of(self, row: str) -> str:
        return f"limit_{row}_rhs_{self.k}"

    def context(self, row: str, coordinate: str) -> RowContext:
        """The row law's context of row ``row`` (``lower`` or ``upper``) of the coordinate ``coordinate``: a unilateral
        row of a known kind (:data:`.sweep.JOINT_LIMIT`) on :class:`LimitZ`."""
        return RowContext(
            row=row,
            kind=str(ROW_JOINT_LIMIT),
            operator=LimitZ(coordinate, f"limit_scale_{self.k}", upper=row == "upper"),
            diagonal=Diagonal(lambda _: "limit_denom"),
            load=self.load,
            store=self.store,
            rhs=self.rhs_of,
            lanes=self.lanes,
            owned=False,
            friction=False,
        )


def _diagonal_limits(dofs: int, dense_dofs: int, laws: RowLaws, lanes: int = 32) -> tuple[str, str, str, str]:
    """The fused joint limits of a world's diagonal-mass coordinates: declarations, loads, the pass and the stores.

    Coordinate ``d`` has a lower and an upper unilateral row (:class:`LimitRows`, :class:`LimitZ`), which the lane
    that owns ``d = lane + lanes k`` steps by the row law (:func:`.sweep.row_update`, never weighted) where the
    coordinate is active, its impulses starting at zero and written back at the end. The dense articulation's
    ``dense_dofs`` coordinates (from ``dense_offset``) hold no fused limit.
    """
    limit_laws = dataclasses.replace(laws, regularize=False, material=False, torsion=False)
    declarations, loads, rows, stores = [], [], [], []
    for k in range((dofs + lanes - 1) // lanes):
        d = f"lane + {lanes * k}" if k else "lane"
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
        limits = LimitRows(k)
        sides = []
        for bit, side in ((1, "lower"), (2, "upper")):
            ctx = limits.context(side, d)
            sides.append(
                f"            if ((limit_active_{k} & {bit}) != 0) {{\n"
                + ctx.operator.dot((("limit_velocity", "row", None),), {"row": side}, " " * 16)
                + "\n"
                + row_update(ctx, limit_laws, residual=ctx.operator.term("limit_velocity"), pad=" " * 16)
                + "\n            }"
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


def slot_rows(dense: int, rows: int, lanes: Lanes) -> SparseFactor:
    """The sparse diagonal's rows over a world's dense group of ``dense`` coordinates plus two slots, a lane an entry.

    The sweep holds the velocity change in factor coordinates: ``u = R^-T L^T P dv`` on the dense articulation (the
    walk's factor and its slips' release) and ``u = sqrt(m) dv`` on a diagonal coordinate (a star's factor is
    diagonal). Row ``i``'s entry ``e < dense`` is its walk's footprint at ``dense_Z[(dense_group * rows + i) * dense +
    e]`` on coordinate ``dense_offset + e``; entries ``dense`` and ``dense + 1`` are its two slots, coordinate
    ``sparse_row_dof[(row_base + i) * 2 + s]`` (``-1`` for none, below ``-1`` a scalar contact's mark) and ``Z_s =
    J_s / sqrt(m_s)``, a key's one step, at ``sparse_row_z``. A contact's rows act on its two bodies' coordinates,
    which the build puts in the same slot of each row (:func:`.walk_rows.slot_header`): its rows share a support.
    """

    def index(row, entry):
        return (
            f"({entry} < {dense} ? dense_offset + {entry} : sparse_row_dof.data[(row_base + {row}) * 2 + {entry} - "
            f"{dense}])"
        )

    def factor(row, entry):
        return (
            f"({entry} < {dense} ? dense_Z.data[(dense_group * {rows} + {row}) * {dense} + {entry}] : "
            f"sparse_row_z.data[(row_base + {row}) * 2 + {entry} - {dense}])"
        )

    return SparseFactor(dense + 2, index, factor, lanes, velocity=Velocity("s_v"))


SCALAR_ROWS = SparseFactor(
    1,
    lambda row, entry: "scalar_coord",
    lambda row, entry: (
        f"sparse_row_z.data[(row_base + {row}) * 2 + (sparse_row_dof.data[(row_base + {row}) * 2] >= 0 ? 0 : 1)]"
    ),
    Lanes(1),
    velocity=Velocity("s_v"),
)
"""An independent scalar contact's rows, one thread: one entry, on its coordinate ``scalar_coord``, through the slot
that holds it (the other is marked); its dense part is empty."""


@dataclasses.dataclass(frozen=True)
class SpeculativeUnits:
    """The serial contact units in their schedule's order (``sparse_contact_serial_normals``), each entered at its head
    and walked row by row, ``i`` at a time: a batch of ``contacts`` units is tested together (``test``, a lane segment a
    unit, declares ``noop_lanes``) and skipped up to its first unit that would change; a unit that ``owns`` a tangent
    pair spans ``rows`` rows and its tangent unit is ``tangent_unit``."""

    capacity: int
    contacts: int
    shift: int
    test: str
    owns: str
    rows: str

    def wrap(self, visit: str, pad: str) -> str:
        head = f"""const int serial_unit_count = sparse_contact_serial_count.data[world];
int serial_unit = 0;
int unit_end = contact_start;
int tangent_unit = -1;
for (int i = 0; i < row_count; ++i) {{
    if (i >= unit_end) {{
        int unit_head = -1;
        while (serial_unit < serial_unit_count) {{
            if ((serial_unit & {self.contacts - 1}) == 0) {{
{indent(self.test, "                ")}
                if (noop_lanes == MASK) {{
                    serial_unit += {self.contacts};
                    continue;
                }}
                serial_unit += FIRST(~noop_lanes) >> {self.shift};
            }}
            unit_head = sparse_contact_serial_normals.data[world * {self.capacity} + serial_unit];
            ++serial_unit;
            break;
        }}
        if (unit_head < 0) break;
        i = unit_head;
        unit_end = i + 1;
        tangent_unit = -1;
        if ({self.owns}) {{
            unit_end = i + {self.rows};
            tangent_unit = i;
        }}
    }}"""
        return indent(head, pad) + f"\n{visit}\n{pad}}}"


def sparse_diagonal_source(sweep: DiagonalSweep) -> tuple[str, Staging]:
    """The native body of the sparse-diagonal Gauss-Seidel sweep ``sweep`` (:class:`DiagonalSweep`'s source): one warp
    per world (or the CPU's one thread, :attr:`DiagonalSweep.lanes`, whose batches hold one contact).

    Every sweep runs the world's fused diagonal joint limits (one lane per coordinate), then its independent scalar
    contacts (a level at a time, one lane per contact: a contact whose coordinate no coupled row touches commutes with
    the coupled rows, and its one-product residual rounds as the warp's sum of it does), then the coupled rows in row order with
    :func:`slot_rows`' warp layout, all on the velocity change in factor coordinates, from zero, or with ``seeded``
    from the response of the impulses the rows carry (:func:`.sweep.impulse_seed`: the warm start's; the fused limits
    start from zero), left once before it is added to the predictor. The exact level law (``levels``) schedules the
    contacts (:func:`.sweep_levels._sparse_contact_levels_kernel`). Every schedule solves a loaded coupled contact unit
    by the unit law (:func:`.sweep.contact_unit`) and visits an unloaded one, and a level's contacts a lane each. With
    ``speculative_contact_batches`` the coupled contacts come from the world's serial list, and a batch of them
    (one per segment of the warp) is tested first: a batch whose normals stay zero without load is skipped whole,
    otherwise the sweep enters the batch at its first active contact. With ``velocity_limits`` the coupled visits
    admit no velocity-limit row, which takes the last word of every sweep (:func:`.sweep.velocity_limit_pass`), as the
    general sweep's; a key's limit row is one step to the world in a slot, as a contact's seed. Under contact ``torsion`` a friction contact's
    spin row follows its pair (four rows: a triple's stride is four); such a contact declines the unit, and its pair's
    visit solves the triple from its metric cached for the launch (:func:`.sweep.friction_metric_law`). The dense
    articulation's change goes to the step's one leave in the walk's coordinates
    (:func:`.factor.walk.walk_leave_kernel`).

    Returns the body and its staging.
    """
    M, D, P = int(sweep.max_constraints), int(sweep.max_world_dofs), int(sweep.dense_dofs)
    torsion, solve_policy = sweep.torsion, sweep.solve_policy
    speculative_contact_batches, velocity_limits = sweep.speculative_contact_batches, sweep.velocity_limits
    L = sweep.lanes
    batch_lanes = min(8 if P + 2 <= 8 else 16, L)
    batch_shift = batch_lanes.bit_length() - 1
    batch_contacts = L // batch_lanes
    scheduled = sweep.levels.law == EXACT

    warp = Lanes(L)
    laws = sweep.laws
    # Every lane (or a scalar contact's one thread) stores what it computed and reads its own stores.
    rows = WorldRows(M, lanes=warp, every_lane=True)
    staging = Staging((Buffer("s_v", "float", D), *rows.buffers()), threads=L)
    # A friction triple reads its metric cached for the launch (sweep.friction_metric_law).
    cached = "ROW_TRIPLE" if torsion else None

    def visit(pad: str, *, scalar: bool = False, kind: int | None = None, parent: str | None = None) -> str:
        """Row ``i``'s row step (:func:`.sweep.row_step`) in a layout: the world's group's (``warp``), or, ``scalar``,
        one thread's (an independent scalar contact on ``scalar_coord``). A ``kind`` (and a friction row's ``parent``)
        known where the visit is emitted replaces the row's metadata."""
        lanes = Lanes(1) if scalar else warp
        store = dataclasses.replace(rows, lanes=lanes)
        op = SCALAR_ROWS if scalar else slot_rows(P, M, lanes)
        # Velocity-limit rows wait for the last word.
        admission = f"if (row_type == {ROW_JOINT_VELOCITY_LIMIT}) continue;" if velocity_limits and kind is None else ""
        return row_step(
            store, op, laws, end="row_count", kind=kind, parent=parent, admission=admission, metric=cached, pad=pad
        )

    pair_op = slot_rows(P, M, warp)
    # A unit's three rows on their shared support: their velocity terms, the pair's couplings with the normal and with
    # each other, in one ladder; the support, the velocity there and the rows' entries kept for the unit's apply.
    unit_sums = (
        ("unit_jv0", "normal", None),
        ("unit_jv1", "tangent1", None),
        ("unit_jv2", "tangent2", None),
        ("gram_n1", "normal", "tangent1"),
        ("gram_n2", "normal", "tangent2"),
        ("unit_cross", "tangent1", "tangent2"),
    )
    unit_terms = pair_op.dot(
        unit_sums, {"normal": "head", "tangent1": "head + 1", "tangent2": "head + 2"}, "        ", keep_velocity=True
    )
    # The unit's changes in one pass on each lane's coordinate, summed before they are added; the lanes synchronize.
    changes = (("normal", "unit_dn"), ("tangent1", "unit_dt1"), ("tangent2", "unit_dt2"))
    unit_apply = (
        f"        if ({' || '.join(f'{change} != 0.0f' for _, change in changes)}) {{\n"
        f"{pair_op.axpy(changes, '            ', kept=True, block=True)}\n        }}\n        {warp.sync()}"
    )

    def contact_rows(head: str) -> str:
        """The rows of the friction contact headed by row ``head``: three, or four with a spin row (the data's,
        :func:`.sweep.spin_row`)."""
        return f"({rows.spin_of(head, 'row_count')} >= 0 ? 4 : 3)" if torsion else "3"

    # A contact unit is a normal and its pair: a contact with a spin row declines it, and its rows take their visits,
    # the triple from its cached metric.
    unit_rows = contact_unit(
        RowContext.of(rows, UnitZ(), end="row_count", metric=cached),
        dataclasses.replace(laws, torsion=False),
        rows="unit_rows",
        declare=f"const int i = head + r;\nconst int row_type = r == 0 ? {ROW_CONTACT} : {ROW_FRICTION};\n"
        f"const float denom = {rows.diag('i')};",
        parent="head",
        check="denom",
        pad="    ",
    )
    loaded = " || ".join(f"{rows.peek(row)} != 0.0f" for row in ("i", "i + 1", "i + 2"))

    def coupled(condition: str, advance: str = "i") -> str:
        """At row ``i``, when ``condition`` holds (``i`` heads a coupled contact unit with a pair, and no spin row) and
        the unit is loaded, the unit on its Gram block in the warp's layout; the loop then moves
        ``advance`` past its rows. An unloaded unit, or a contact with a spin row, falls through to the loop's row
        visits."""
        spinless = f" && {rows.spin_of('i', 'row_count')} < 0" if torsion else ""
        text = f"""if ({condition}{spinless}) {{
    const int unit_rows = 3;
    if ({loaded}) {{
        const int head = i;
{unit_terms}
{indent(unit_rows, "    ")}
{unit_apply}
        {advance} += unit_rows - 1;
        continue;
    }}
}}"""
        return indent(text, " " * 12)

    def independent(head: str) -> str:
        """Whether the contact headed by row ``head`` is an independent scalar contact (a slot holds its mark)."""
        return f"sparse_row_dof.data[(row_base + {head}) * 2] < -1 || sparse_row_dof.data[(row_base + {head}) * 2 + 1] < -1"

    def owns_tangents(head: str, pad: str) -> str:
        """Whether unit head ``head`` owns a tangent pair: the row after it is a tangent naming it as parent."""
        return (
            f"{head} + 2 < row_count\n"
            f"{pad}&& ({rows.kind(f'{head} + 1')}) == {ROW_FRICTION}\n"
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

    # Independent scalar contacts: a level at a time, one lane per contact. Schedule construction excludes every
    # coordinate a coupled row touches, so they commute with the coupled rows, and a coordinate's contacts lie in
    # successive levels in row order. A contact is a normal row and, for a triple or a unit that owns an anchor, the
    # tangent pair after it, which names the normal as parent: the visits' kinds are known here, so each compiles to
    # its own law. The pair is solved at its first tangent. A visit that skips its row leaves its block (do-while),
    # not the level.
    independent_phase = ""
    if scheduled:
        pad = " " * 16
        tangents = f"""{pad}do {{
{pad}    const int i = head + 1;
{visit(pad + "    ", scalar=True, kind=ROW_FRICTION, parent="head")}
{pad}}} while (false);"""
        tangents = f"{pad}if ({owns_tangents('head', pad + '    ')}) {{\n{indent(tangents, '    ')}\n{pad}}}"
        normal = f"""                do {{
                    const int i = head;
{visit(pad + "    ", scalar=True, kind=ROW_CONTACT)}
                }} while (false);
{tangents}"""
        # A level holds at most one contact a coordinate, and a coordinate's contacts lie in successive levels in row
        # order; a contact's coordinate (scalar_coord) is its unmarked slot's.
        levels = Levels(
            "sparse_contact_level_count.data[world]",
            lambda level: f"sparse_contact_level_offsets.data[world * {M + 1} + {level}]",
            f"""const int head = sparse_contact_level_units.data[world * {M} + place];
const int normal_coord_0 = sparse_row_dof.data[(row_base + head) * 2];
const int normal_coord_1 = sparse_row_dof.data[(row_base + head) * 2 + 1];
const int scalar_coord = normal_coord_0 >= 0 ? normal_coord_0 : normal_coord_1;""",
            stride=L,
        )
        independent_phase = "\n" + levels.wrap(normal, "        ") + "\n"

    def batch(index: str, count: str) -> str:
        """Test the batch of serial contacts from ``index``: one contact per segment of ``batch_lanes`` lanes, its
        normal's prospective update; declares ``noop_lanes``, the lanes whose contact would change nothing."""
        segment = slot_rows(P, M, Lanes(batch_lanes, "batch_lane", segment=True))
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
{segment.dot((("candidate_velocity", "row", None),), {"row": "candidate_normal"}, "", exists={"row": "candidate_valid"})}
bool candidate_noop = !candidate_valid;
if (candidate_valid) {{
    const float old_impulse = {rows.peek("candidate_normal")};
    float new_impulse = old_impulse;
    if ({rows.diag("candidate_normal")} > 0.0f)
        new_impulse = ProjectionLaw::unilateral(ProjectionLaw::relaxed_step(old_impulse, {step}, omega));
    {noop}
}}
const unsigned noop_lanes = BALLOT(candidate_noop);"""

    row = visit("            ")
    if not scheduled:
        # The rows hold no schedule: a normal row owning the tangent pair after it heads a contact unit, unless a
        # mark makes it a scalar contact (its rows keep their visits, as a scheduled sweep's level phase visits them).
        head = (
            f"({rows.kind('i')}) == {ROW_CONTACT} && {owns_tangents('i', ' ' * 16)}\n{' ' * 16}&& !({independent('i')})"
        )
        serial = SlotOrder("row_count").wrap(f"{coupled(head)}\n{row}", "        ")
    elif not speculative_contact_batches:
        serial = SlotOrder("row_count").wrap(
            f"""            if (i >= contact_start) {{
                // A tangent row's parent is its unit's head.
                const int unit_head = ({rows.kind("i")}) == {ROW_FRICTION} ? {rows.parent("i")} : i;
                if ({independent("unit_head")}) continue;
            }}
{coupled(f"i >= contact_start && ({rows.kind('i')}) == {ROW_CONTACT} && {owns_tangents('i', ' ' * 16)}")}
{row}
{inactive_pair(f"i >= contact_start && ({rows.kind('i')}) == {ROW_CONTACT} && {owns_tangents('i', ' ' * 16)}", "i")}""",
            "        ",
        )
    else:
        # Units have one or three rows (four with a spin row): the loop walks real rows and enters each serial unit at
        # its head.
        units = SpeculativeUnits(
            M,
            batch_contacts,
            batch_shift,
            batch("serial_unit", "serial_unit_count"),
            owns_tangents("i", " " * 12),
            contact_rows("i"),
        )
        serial = units.wrap(
            f"""{coupled("i == tangent_unit")}
{row}
{inactive_pair("i == tangent_unit", "i")}""",
            "        ",
        )

    declarations, limit_loads, limit_rows, limit_stores = _diagonal_limits(D, P, laws, L)
    # A diagonal coordinate's factor is sqrt(m): its scale m^-1/2 (zero on the dense articulation's coordinates and
    # where no coordinate is mapped, which keep their velocity).
    diagonal_scale = (
        "const bool dense_coordinate = d >= dense_offset && d < dense_offset + "
        f"{P};\nconst float diagonal_scale = !dense_coordinate && global_dof >= 0\n"
        "    ? sqrtf(diagonal_inverse_mass.data[global_dof]) : 0.0f;"
    )
    sweep_end = end_sweep(solve_policy, iteration=ITERATION)
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
    # The impulses the rows carry into the first sweep, row by row in slot order through both slots and the dense part
    # (a marked slot names no coordinate), as the general sweep's seed.
    seed = ""
    if sweep.seeded:
        apply = pair_op.axpy((("row", "seed"),), "        ", row="i")
        seed = impulse_seed("row_count", rows.peek("i"), f"({rows.kind('i', scan=True)})", apply, warp.sync()) + "\n"
    last_word = velocity_limit_pass(rows, pair_op, laws, end="row_count", pad="        ") if velocity_limits else ""
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

    for (int i = lane; i < row_count; i += {L}) {{
{indent(rows.loads("i"), "        ")}
    }}
    for (int d = lane; d < {D}; d += {L}) s_v[d] = 0.0f;
{limit_loads}
    SYNC();
{seed}{metric}
{level_loop((limit_rows, "        SYNC();", independent_phase, serial), (last_word,), exit=sweep_end)}

    for (int i = lane; i < row_count; i += {L}) world_impulses.data[off_dense + i] = {rows.peek("i")};
    SYNC();
    // Leave factor coordinates: the dense articulation's change in the walk's coordinates to the step's one leave
    // (tree_factor.walk_leave_kernel, dv = P^T L^-T R^T u); dv = u / sqrt(m) on a diagonal coordinate.
    for (int d = lane; d < {D}; d += {L}) {{
        const int global_dof = world_dof_indices.data[dof_map_base + d];
{indent(diagonal_scale, "        ")}
        if (global_dof < 0) continue;
        if (dense_coordinate)
            walk.du.data[global_dof] += s_v[d];
        else
            v_out.data[global_dof] += diagonal_scale > 0.0f ? s_v[d] * diagonal_scale : s_v[d];
    }}
{limit_stores}
"""
    return sweep.native(body, lanes=L), staging


@dataclasses.dataclass(frozen=True)
class DiagonalSweep(SweepConfig):
    """The sparse-diagonal sweep's configuration (case K2): a world of a dense articulation of ``dense_dofs``
    coordinates beside one-coordinate trees on a fixed frame, ``max_world_dofs`` in all, at most ``max_constraints``
    rows (:func:`sparse_diagonal_source`).

    Its switches, each the plan's from the class's shape and the laws (:meth:`.sweep_plan.Sweep.build`):

    - the support classes: the arm's rows on its dense group and two one-coordinate slots, a key's rows one entry
      (the class plan's diagonal rule);
    - the levels: under the exact law (``levels``, :class:`.sweep.LevelLaw`) the independent scalar contacts a level
      at a time, a lane a contact (the diagonal choice's ``units``); with
      ``speculative_contact_batches`` its serial contacts tested a batch at a time (at most 16 response entries);
    - the fused limits: each diagonal coordinate's joint limits, a lane a coordinate, exact;
    - ``seeded``: the first sweep starts from the response of the impulses the rows carry;
    - ``velocity_limits``: the velocity-limit rows take the last word of every sweep.

    A world is a warp on CUDA and one thread on the CPU (:meth:`work`); weighted rows
    (:func:`.sweep.weight_arguments`) and material contacts (:func:`.sweep.material_arguments`) add their operands.
    """

    max_constraints: int
    max_world_dofs: int
    dense_dofs: int
    device_arch: str
    _: dataclasses.KW_ONLY
    speculative_contact_batches: bool = False
    seeded: bool = False
    velocity_limits: bool = False

    ABI: ClassVar[tuple] = SPARSE_DIAGONAL_ABI
    NATIVE: ClassVar[str] = "sweep_sparse_diagonal_native"
    LABEL: ClassVar[str] = "the sparse-diagonal sweep"

    def __post_init__(self):
        if self.levels.law not in (SEQUENTIAL, EXACT):
            raise ValueError("the sparse diagonal's contacts step in order or by exact levels")
        if self.speculative_contact_batches and (self.levels.law != EXACT or int(self.dense_dofs) + 2 > 16):
            raise ValueError("speculative contact batches require exact levels with at most 16 response entries")

    @property
    def lanes(self) -> int:
        """A world's lanes: a warp on CUDA, the CPU's one thread (:func:`.execution.device_lanes`), whose batches hold
        one contact."""
        return device_lanes(self.device_arch, 32)

    @property
    def name(self) -> str:
        return (
            f"sweep_sparse_diagonal_{int(self.max_constraints)}_{int(self.max_world_dofs)}_{int(self.dense_dofs)}"
            + ("_speculative_batches" if self.speculative_contact_batches else "")
            + ("_units" if self.levels.law == EXACT else "")
            + ("_weighted" if self.weighted else "")
            + ("_material" if self.material else "")
            + ("_torsion" if self.torsion else "")
            + ("_seeded" if self.seeded else "")
            + ("_velocity_limits" if self.velocity_limits else "")
            + ("" if self.lanes == 32 else f"_lanes{self.lanes}")
            + self.solve_policy.suffix
        )

    source = sparse_diagonal_source

    def dispatch(self, staging: Staging) -> Dispatch:
        return kernel_template.items("world")

    def operands(self) -> tuple[RowArgument, ...]:
        return (
            # Where spin rows may appear, every friction triple's metric (sweep.friction_metric_stage).
            *((RowArgument("friction_metric", wp.array3d[float], after="omega"),) if self.torsion else ()),
            *self.law_operands(),
        )

    def work(self, items: int = 0, shared: int = 0) -> Work:
        """A block a world, a warp (one thread a world on the CPU); no register line, so no launch bound."""
        return Work(int(items), lanes=32, block=32, shared=int(shared))

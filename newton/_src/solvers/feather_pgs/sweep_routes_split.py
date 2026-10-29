# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Split mode's loop sweeps as configurations of the sweep emitter (:mod:`.sweep`), one native source for both devices.

One thread sweeps one world in place (``wp.launch(dim=world_count)``): the world's impulses and, for free bodies, its
velocities stay in global memory, and every row takes the row law (:func:`.sweep.row_visit`) on one thread's layout of
the world's response operator.

- :func:`loop_kernel`: the assembled Delassus rows (:class:`.Assembled`), in impulse space;
- :func:`mf_loop_kernel`: the free-body rows (:class:`.FreeBodyZ` in the loop's order), velocities through the body
  map (``body_to_articulation``, ``art_dof_start``) in factor coordinates, each body's speed row in row order;
- :func:`loop_unit_kernel`, :func:`mf_loop_unit_kernel`: the same sweeps with exact contact units
  (``contact_update="unit"``): once friction is active, a contact unit (a point contact, or a patch ring as one ring
  unit or as anchor units) is solved at its first row by the contact-unit law (:mod:`.contact_unit`), every other
  row as above.

The kernels are dual-native (:func:`.device_preludes.dual_native`): the projection laws, the patch ring's walks and the
contact-unit law have CPU twins, and one thread needs no lane group.
"""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Callable

import numpy as np
import warp as wp

from . import kernel_template
from .contact_unit import (
    CONTACT_UNIT,
    CONTACT_UNIT_NEWTON_CAP,
    CONTACT_UNIT_PROXIMAL,
    CONTACT_UNIT_RING_ANCHORS,
    CONTACT_UNIT_RING_NORMALS,
    CONTACT_UNIT_RING_PASSES,
    RING_ALPHA,
    RING_BLOCK,
    RING_BLOCK_SIZE,
    RING_IMPULSE,
    RING_RESIDUAL,
    RING_SOLUTION,
)
from .device_preludes import RING
from .kernel_template import KernelTemplate
from .kernels import BodyFactors
from .law_source import law_preludes
from .projection_laws import PROJECTION_LAW
from .response_operator import Assembled, FreeBodyZ, Lanes, ResponseOperator, Velocity
from .solve_policy import DEFAULT_POLICY, Change, SolvePolicy, bound, sweep_exit
from .staging import Staging
from .sweep import (
    CONTACT,
    CPU_MATH,
    FRICTION,
    JOINT_LIMIT,
    VELOCITY_LIMIT,
    RowContext,
    RowLaws,
    change_vote,
    indent,
    row_update,
    row_visit,
)
from .sweep import speed_row as speed_row_update

# ---------------------------------------------------------------------------------------------------------------------
# Operands
# ---------------------------------------------------------------------------------------------------------------------

LOOP_ABI = kernel_template.params(
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
"""The assembled loop sweep's operands (one world per thread); the arrays' row capacity is their second extent."""

UNIT_ABI = kernel_template.params(
    ("ring_units", int),
    ("unit_state", wp.array3d[float]),
    ("unit_rows", wp.array2d[int]),
    ("unit_block", wp.array[float]),
)
"""The contact units' switch and per-world storage: ``ring_units`` (1: a patch ring is one ring unit, 0: anchor
units), ``unit_state`` (``[world, row, 3]``: :data:`UNIT_STATE`), ``unit_rows`` (``[world, row]``, a ring's rows) and
``unit_block`` (:data:`.contact_unit.RING_BLOCK_SIZE` floats per world, a ring unit's block)."""

LOOP_UNIT_ABI = LOOP_ABI + UNIT_ABI
"""The assembled loop sweep's operands with contact units."""

_MF_ROWS = kernel_template.params(
    ("mf_constraint_count", wp.array[int]),
    ("mf_body_a", wp.array2d[int]),
    ("mf_body_b", wp.array2d[int]),
    ("mf_Z_a", wp.array3d[float]),
    ("mf_Z_b", wp.array3d[float]),
    ("factors", BodyFactors),
    ("mf_eff_mass_inv", wp.array2d[float]),
    ("mf_rhs", wp.array2d[float]),
    ("mf_row_type", wp.array2d[int]),
    ("mf_row_parent", wp.array2d[int]),
    ("mf_row_mu", wp.array2d[float]),
    ("mf_row_w", wp.array2d[float]),
    ("body_to_articulation", wp.array[int]),
    ("art_dof_start", wp.array[int]),
    ("iterations", int),
    ("omega", float),
    ("regularize", int),
)
_MF_SCHEDULE = kernel_template.params(("friction_start_iteration", int), ("iteration_offset", int))
_MF_STATE = kernel_template.params(("mf_impulses", wp.array2d[float]), ("v_out", wp.array[float]))

MF_LOOP_ABI = _MF_ROWS + _MF_SCHEDULE + _MF_STATE
"""The free-body loop sweep's operands (one world per thread)."""

MF_LOOP_UNIT_ABI = _MF_ROWS + _MF_SCHEDULE + UNIT_ABI + _MF_STATE
"""The free-body loop sweep's operands with contact units."""

UNIT_STATE = ("SLIDE_ROOT", "LOAD_CHANGE", "LATCH_FACTOR")
"""A row's unit state (``unit_state[world, row]``): the slide root of the unit or anchor whose normal is the row, and
the latch (its last applied load change and relaxation factor) of the unit whose first row it is."""

_PRELUDES = (CPU_MATH, RING, *law_preludes(PROJECTION_LAW))
"""The loop sweeps' device preludes: each has a CPU twin."""
_UNIT_PRELUDES = (*_PRELUDES, CONTACT_UNIT)
"""With the contact-unit law, whose CPU twin is its CPU binding (Warp's helpers), the one its ``wp.func_native``
functions take."""


# ---------------------------------------------------------------------------------------------------------------------
# A world's rows in place, and the representations the loop sweeps read them in
# ---------------------------------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class LoopRows:
    """A world's rows as a loop sweep reads them in place: row ``r`` of each per-row array at ``offset + r``.

    The impulses are read and written through ``lam``, the world's first impulse; a patch ring's walk stops after
    ``capacity`` rows (the arrays' row capacity).
    """

    offset: str
    types: str
    parents: str
    rhs_array: str
    mu_array: str
    weight_array: str

    def at(self, array: str, row: str) -> str:
        return f"{array}.data[{self.offset} + {row}]"

    def kind(self, row: str) -> str:
        return self.at(self.types, row)

    def parent(self, row: str) -> str:
        return self.at(self.parents, row)

    def load(self, row: str) -> str:
        return f"lam[{row}]"

    def store(self, row: str, value: str) -> str:
        return f"lam[{row}] = {value};"

    def rhs(self, row: str) -> str:
        return self.at(self.rhs_array, row)

    def mu(self, row: str) -> str:
        return self.at(self.mu_array, row)

    def weight(self, row: str) -> str:
        return self.at(self.weight_array, row)

    def ring_load(self, parent: str) -> str:
        """The pooled normal load of the patch ring of normal row ``parent``."""
        return (
            f"RING_LOAD(lam[{parent}], {parent}, capacity,\n"
            f"    [&](int r) {{ return {self.parent('r')}; }}, [&](int r) {{ return lam[r]; }})"
        )


@dataclasses.dataclass(frozen=True)
class LoopSweep:
    """A loop sweep's representation of a world's rows: its row store, its operator and how a row is read.

    Args:
        count: The world's row counts.
        impulses: The rows' impulses.
        rows: The row store.
        operator: The response operator, one thread's layout.
        kind: The visited row's type, a local.
        declare: Declares row ``i``'s stored diagonal (or inverse diagonal), which must be positive, under the name
            the operator reads it by.
        check: That name.
        bodies: Declares a row's bodies (``row``, the names of their first coordinates), when the operator reads
            them; ``None`` for an operator in impulse space.
        contact_kinds: The unilateral row kinds.
        pair_residual: A friction pair reduces its first tangent's residual again (:attr:`.RowContext.pair_residual`).
        setup: Declarations of the world's operands, and what precedes the sweeps.
        unit_diagonal: Statements returning a unit block's diagonal entry of row ``r``.
        unit_entry: Statements returning a unit block's entry of rows ``r`` and ``c``.
        finish: What follows the sweeps.
    """

    count: str
    impulses: str
    rows: LoopRows
    operator: ResponseOperator
    kind: str
    declare: str
    check: str
    bodies: Callable[[str, tuple[str, str]], str] | None
    contact_kinds: tuple[int, ...]
    pair_residual: bool
    setup: str
    unit_diagonal: str
    unit_entry: str
    finish: str = ""

    def context(self, *, kind: str | None = None, friction: bool = True) -> RowContext:
        rows = self.rows
        return RowContext(
            row="i",
            kind=kind or self.kind,
            operator=self.operator,
            load=rows.load,
            store=rows.store,
            rhs=rows.rhs,
            mu=rows.mu,
            weight=rows.weight,
            parent="parent",
            ring_load=rows.ring_load,
            lanes=Lanes(1),
            contact_kinds=self.contact_kinds,
            friction=friction,
            pair_residual=self.pair_residual,
        )

    def entries(self, row: str = "i", names: tuple[str, str] = ("dof_a", "dof_b")) -> str:
        """Declare the bodies of row ``row`` (nothing in impulse space)."""
        return "" if self.bodies is None else self.bodies(row, names)

    def residual(self, out: str, pad: str) -> str:
        """``float out``: row ``i``'s velocity term as the operator sums it, its bodies declared first."""
        entries = self.entries()
        return (indent(entries, pad) + "\n" if entries else "") + self.operator.residual(out, pad)


def _dense() -> LoopSweep:
    """The assembled Delassus rows: ``C`` in global memory, row ``i``'s sums from its right-hand side."""
    rows = LoopRows("off1", "world_row_type", "world_row_parent", "world_rhs", "world_row_mu", "world_row_w")
    stored = functools.partial(rows.at, "world_diag")
    op = Assembled(
        0,
        "capacity",
        "global",
        Lanes(1),
        impulse="lam",
        count="m",
        rhs=rows.rhs,
        diagonal=lambda row: "denom" if row == "i" else stored(row),
    )
    return LoopSweep(
        count="world_constraint_count",
        impulses="world_impulses",
        rows=rows,
        operator=op,
        kind="row_type",
        declare=f"const float denom = {stored('i')};",
        check="denom",
        bodies=None,
        contact_kinds=(CONTACT, JOINT_LIMIT, VELOCITY_LIMIT),
        pair_residual=False,
        setup="const int off2 = off1 * capacity;",
        # A unit's diagonal is the larger of the stored diagonal and C's own entry.
        unit_diagonal=f"{op.coupling('entry', 'r', 'r', '')}\nreturn fmaxf({op.value('r')}, entry);",
        unit_entry=f"{op.coupling('entry', 'r', 'c', '')}\nreturn entry;",
    )


def _free_body() -> LoopSweep:
    """The free-body rows in the loop's order: velocities in place, each body at its articulation's coordinates, in
    its factor coordinates between the sweeps' entry and exit (each body converted once, at its first row)."""
    rows = LoopRows("c_off", "mf_row_type", "mf_row_parent", "mf_rhs", "mf_row_mu", "mf_row_w")
    stored = functools.partial(rows.at, "mf_eff_mass_inv")
    op = FreeBodyZ(
        Lanes(1),
        velocity=Velocity("v_out.data"),
        inverse=lambda row: "mf_diag" if row == "i" else stored(row),
        order="loop",
        factor=lambda first: f"factors.dof_factor.data[{first}]",
    )

    def convert(*, enter: bool) -> str:
        action = op.enter if enter else op.leave
        return f"""for (int r = 0; r < m; ++r) {{
    for (int side = 0; side < 2; ++side) {{
        if (!FIRST_SEEN(r, side)) continue;
        const int first = BODY_DOF(side == 0 ? {rows.at("mf_body_a", "r")} : {rows.at("mf_body_b", "r")});
        const int factor = factors.dof_factor.data[first];
{indent(action(lambda k: f"v_out.data[first + {k}]", "factor"), "        ")}
    }}
}}"""

    def bodies(row: str, names: tuple[str, str]) -> str:
        return "\n".join(
            f"const int {name} = BODY_DOF({rows.at(array, row)});"
            for name, array in zip(names, ("mf_body_a", "mf_body_b"), strict=True)
        )

    entry = op.shared_coupling("entry", "r", "c", ("column_a", "column_b"), "")
    return LoopSweep(
        count="mf_constraint_count",
        impulses="mf_impulses",
        rows=rows,
        operator=op,
        kind="mf_rt",
        declare=f"const float mf_diag = {stored('i')};",
        check="mf_diag",
        bodies=bodies,
        contact_kinds=(CONTACT,),
        pair_residual=True,
        setup=(
            "const int mf6_base = c_off * 6;\n"
            "// A body's first velocity coordinate, its articulation's (-1 for no body).\n"
            "const auto BODY_DOF = [&](const int body) -> int {\n"
            "    return body >= 0 ? art_dof_start.data[body_to_articulation.data[body]] : -1;\n"
            "};\n"
            "// Whether side `side` (0: a, 1: b) of row `r` holds its body's first appearance in row order.\n"
            "const auto FIRST_SEEN = [&](const int r, const int side) -> bool {\n"
            f"    const int body = side == 0 ? {rows.at('mf_body_a', 'r')} : {rows.at('mf_body_b', 'r')};\n"
            "    if (body < 0) return false;\n"
            "    for (int q = 0; q < r; ++q)\n"
            f"        if ({rows.at('mf_body_a', 'q')} == body || {rows.at('mf_body_b', 'q')} == body) return false;\n"
            f"    return side == 0 || {rows.at('mf_body_a', 'r')} != body;\n"
            "};\n"
            "// The world's bodies enter factor coordinates.\n" + convert(enter=True)
        ),
        unit_diagonal=f"return {op.value('r')};",
        unit_entry=f"{bodies('r', ('dof_a', 'dof_b'))}\n{bodies('c', ('column_a', 'column_b'))}\n{entry}\nreturn entry;",
        finish="// The world's bodies leave factor coordinates.\n" + convert(enter=False),
    )


def _laws(solve_policy: SolvePolicy) -> RowLaws:
    """The loop sweeps' row laws under ``solve_policy``."""
    return RowLaws(drive=False, regularize=True, policy=solve_policy)


"""The loop sweeps' row laws: no drive rows; a contact's step takes its regularization weight when ``regularize``."""


# ---------------------------------------------------------------------------------------------------------------------
# Contact units: a unit's rows solved together, at its first row, by the contact-unit law
# ---------------------------------------------------------------------------------------------------------------------


def _float(value: float) -> str:
    """The C++ literal of ``value`` as a float32 (the value Warp passes for a float constant)."""
    return f"{np.float32(value)!s}f"


def _lambda(head: str, body: str) -> str:
    """``const auto head { body };``, the body's empty lines dropped."""
    body = "\n".join(line for line in body.split("\n") if line.strip())
    return f"    const auto {head} {{\n{indent(body, '        ')}\n    }};"


def contact_units(sweep: LoopSweep, laws: RowLaws) -> tuple[str, str]:
    """The contact units of a loop sweep: their visits, declared once per world, and the row loop's admission.

    Once friction is active, a contact normal whose patch ring has several normals or an anchor (a normal its
    tangent pair follows) belongs to a unit, solved at the ring's first row; the unit's tangent rows are not visited.
    A point unit or an anchor solves its 3x3 block (``ContactUnit::solve``) against the residuals with its own
    impulses removed, a ring unit its normals and up to two anchors' pairs together (``ContactUnit::ring``), and a
    ring member without a pair takes the contact row's update. Each unit's alternation latch (``ContactUnit::latch``)
    relaxes the move; a unit that holds nothing and does not approach stays at zero. The operator binds the units'
    arithmetic: a row's whole residual, the block's entries and the apply of a row's change. A unit votes its rows'
    changes under the exit law; its state (latch, slide root) follows the impulses and votes nothing.

    Returns the declarations (after the world's operands, before the sweeps) and the admission (after the row's type
    is read; ``friction`` is set per sweep).
    """
    rows, op = sweep.rows, sweep.operator
    fields = "unit_fields"
    whole = "jv" if op.residual_includes_rhs else f"jv + {rows.rhs('i')}"
    residual = sweep.residual("jv", "")
    scalar_ctx = sweep.context(kind="row_type", friction=False)
    scalar = "\n".join(
        line
        for line in (
            f"const int row_type = {CONTACT};",
            sweep.declare,
            f"if ({sweep.check} <= 0.0f) return;",
            residual,
            row_update(scalar_ctx, laws, residual="jv", pad=""),
        )
        if line
    )

    def applies(*changes: tuple[str, str]) -> str:
        if not op.applies:
            return ""
        return "\n".join(f"UNIT_APPLY({row}, {change});" for row, change in changes)

    def unit_change(row: str, kind: int) -> Change:
        # The point unit's rows: their whole residuals r_<row> and old impulses o_<row> are in scope at the vote.
        rhs = rows.rhs(row)
        unit_bound = bound(rhs, op.terms(f"r_{row}", rhs, whole=True), f"o_{row}", diagonal=f"w_{row}")
        return Change(f"new_{row} - o_{row}", unit_bound, str(kind), f"w_{row}", scaled=True)

    point_vote = (
        "if (new_n != o_n || new_t0 != o_t0 || new_t1 != o_t1) {\n"
        + change_vote(
            laws, (unit_change("n", CONTACT), unit_change("t0", FRICTION), unit_change("t1", FRICTION)), "    "
        )
        + "\n}"
    )
    # A ring member's residual is folded into the block's right-hand side: the member votes on a zero bound of S_i.
    ring_change = Change(
        "updated - old",
        bound(rows.rhs("ring_rows[a]"), "0.0f", "old", diagonal="ring_W[a * rows + a]"),
        f"(a < normals ? {CONTACT} : {FRICTION})",
        "ring_W[a * rows + a]",
        scaled=True,
    )
    ring_vote = indent("if (updated != old) {\n" + change_vote(laws, (ring_change,), "    ") + "\n}", "    ")
    point = f"""const int t0 = n + 1, t1 = n + 2;
const float r_n = UNIT_RESIDUAL(n);
const float o_n = lam[n], o_t0 = lam[t0], o_t1 = lam[t1];
if (o_n == 0.0f && o_t0 == 0.0f && o_t1 == 0.0f && ring_load == 0.0f && r_n >= 0.0f) {{
    // Separated: the exact solution is zero, and so is the latch's load change.
    lam[n] = 0.0f;
    lam[t0] = 0.0f;
    lam[t1] = 0.0f;
    UNIT_KEEP(n, LOAD_CHANGE, 0.0f);
    return;
}}
const float w_n = UNIT_DIAGONAL(n);
// A normal without response cannot move, as the sequential update skips it.
if (w_n <= 0.0f) return;
const float r_t0 = UNIT_RESIDUAL(t0), r_t1 = UNIT_RESIDUAL(t1);
const float w_t0 = UNIT_DIAGONAL(t0), w_t1 = UNIT_DIAGONAL(t1);
const float w_n_t0 = UNIT_ENTRY(n, t0), w_n_t1 = UNIT_ENTRY(n, t1);
const float w_t0_n = UNIT_ENTRY(t0, n), w_t0_t1 = UNIT_ENTRY(t0, t1);
const float w_t1_n = UNIT_ENTRY(t1, n), w_t1_t0 = UNIT_ENTRY(t1, t0);
// The unit's residuals with its own impulses removed, b = r - W_U lambda_U, row by row; the block is unregularized.
const float b_n = r_n - w_n * o_n - w_n_t0 * o_t0 - w_n_t1 * o_t1;
const float b_t0 = r_t0 - w_t0_n * o_n - w_t0 * o_t0 - w_t0_t1 * o_t1;
const float b_t1 = r_t1 - w_t1_n * o_n - w_t1_t0 * o_t0 - w_t1 * o_t1;
// A point unit reads its normal-tangent coupling from the normal row, an anchor from its tangent rows.
const float w_nt0 = anchor != 0 ? w_t0_n : w_n_t0;
const float w_nt1 = anchor != 0 ? w_t1_n : w_n_t1;
const ContactUnitSolution solution = ContactUnit::solve(
    ProjectionLaw::regularized_diagonal(w_n, CONTACT_WEIGHT(n)), w_t0, w_t0_t1, w_t1, w_nt0, w_nt1, b_n, b_t0, b_t1,
    {rows.mu("t0")}, ring_load, STATE(n, SLIDE_ROOT), {CONTACT_UNIT_NEWTON_CAP}, o_n, o_t0, o_t1);
if (solution.alpha > 0.0f) UNIT_KEEP(n, SLIDE_ROOT, solution.alpha);
const ContactUnitLatch latch = ContactUnit::latch(STATE(n, LATCH_FACTOR), STATE(n, LOAD_CHANGE), o_n, solution.normal);
UNIT_KEEP(n, LATCH_FACTOR, latch.factor);
UNIT_KEEP(n, LOAD_CHANGE, latch.change);
const float new_n = o_n + latch.weight * (solution.normal - o_n);
const float new_t0 = o_t0 + latch.weight * (solution.friction0 - o_t0);
const float new_t1 = o_t1 + latch.weight * (solution.friction1 - o_t1);
lam[n] = new_n;
lam[t0] = new_t0;
lam[t1] = new_t1;
{point_vote}
{applies(("n", "new_n - o_n"), ("t0", "new_t0 - o_t0"), ("t1", "new_t1 - o_t1"))}"""

    ring = f"""const int first = ring_rows[0];
int anchors = 0;
int held = 0;
int approaching = 0;
for (int a = 0; a < normals; ++a) {{
    const int row = ring_rows[a];
    if (HAS_PAIR(row)) {{
        ring_rows[normals + 2 * anchors] = row + 1;
        ring_rows[normals + 2 * anchors + 1] = row + 2;
        anchors += 1;
    }}
    const float r = UNIT_RESIDUAL(row);
    ring_b[a] = r;
    if (lam[row] != 0.0f) held = 1;
    if (r < 0.0f) approaching = 1;
}}
const int rows = normals + 2 * anchors;
for (int a = normals; a < rows; ++a) {{
    if (lam[ring_rows[a]] != 0.0f) held = 1;
}}
if (held == 0 && approaching == 0) {{
    // Separated: the exact solution is zero, and so is the latch's load change.
    for (int a = 0; a < rows; ++a) lam[ring_rows[a]] = 0.0f;
    UNIT_KEEP(first, LOAD_CHANGE, 0.0f);
    return;
}}
for (int a = normals; a < rows; ++a) ring_b[a] = UNIT_RESIDUAL(ring_rows[a]);
for (int a = 0; a < rows; ++a) {{
    const int row = ring_rows[a];
    ring_old[a] = lam[row];
    for (int c = 0; c < rows; ++c) ring_W[a * rows + c] = c == a ? UNIT_DIAGONAL(row) : UNIT_ENTRY(row, ring_rows[c]);
}}
// b = r - W_U lambda_U with the unregularized block, which the regularized one leaves unchanged.
for (int a = 0; a < rows; ++a) {{
    float v = ring_b[a];
    for (int c = 0; c < rows; ++c) v -= ring_W[a * rows + c] * ring_old[c];
    ring_b[a] = v;
}}
for (int a = 0; a < normals; ++a) {{
    ring_W[a * rows + a] = ProjectionLaw::regularized_diagonal(ring_W[a * rows + a], CONTACT_WEIGHT(ring_rows[a]));
}}
float mu0 = 0.0f;
float mu1 = 0.0f;
if (anchors >= 1) {{
    mu0 = {rows.mu("ring_rows[normals]")};
    ring_alpha[0] = STATE(ring_rows[normals] - 1, SLIDE_ROOT);
}}
if (anchors >= 2) {{
    mu1 = {rows.mu("ring_rows[normals + 2]")};
    ring_alpha[1] = STATE(ring_rows[normals + 2] - 1, SLIDE_ROOT);
}}
ContactUnit::ring(ring_W, ring_b, ring_old, ring_x, ring_alpha, normals, anchors, mu0, mu1, {CONTACT_UNIT_RING_PASSES},
                  {CONTACT_UNIT_NEWTON_CAP}, {_float(CONTACT_UNIT_PROXIMAL)});
if (anchors >= 1) UNIT_KEEP(ring_rows[normals] - 1, SLIDE_ROOT, ring_alpha[0]);
if (anchors >= 2) UNIT_KEEP(ring_rows[normals + 2] - 1, SLIDE_ROOT, ring_alpha[1]);
float old_load = 0.0f;
float new_load = 0.0f;
for (int a = 0; a < normals; ++a) {{
    old_load += ring_old[a];
    new_load += ring_x[a];
}}
const ContactUnitLatch latch = ContactUnit::latch(STATE(first, LATCH_FACTOR), STATE(first, LOAD_CHANGE), old_load,
                                                  new_load);
UNIT_KEEP(first, LATCH_FACTOR, latch.factor);
UNIT_KEEP(first, LOAD_CHANGE, latch.change);
for (int a = 0; a < rows; ++a) {{
    const float old = ring_old[a];
    const float updated = old + latch.weight * (ring_x[a] - old);
    lam[ring_rows[a]] = updated;
{ring_vote}
{indent(applies(("ring_rows[a]", "updated - old")), "    ")}
}}"""

    has_pair = (
        "// Normal row n is an anchor: its tangent pair follows it.\n"
        f"return n + 2 < m && {rows.kind('n + 1')} == {FRICTION} && {rows.parent('n + 1')} == n;"
    )
    shape = f"""// The normals, the anchors and the first row of normal n's patch ring (or point contact).
normals = 1;
anchors = HAS_PAIR(n) ? 1 : 0;
first = n;
int row = {rows.parent("n")};
int count = 0;
while (row >= 0 && row != n && count < m) {{
    normals += 1;
    if (HAS_PAIR(row)) anchors += 1;
    if (row < first) first = row;
    row = {rows.parent("row")};
    count += 1;
}}"""
    sort = f"""// The ring of its smallest row first, in increasing row order.
ring_rows[0] = first;
int count = 1;
int row = {rows.parent("first")};
int guard = 0;
while (row >= 0 && row != first && guard < m) {{
    int position = count;
    while (position > 0 && ring_rows[position - 1] > row) {{
        ring_rows[position] = ring_rows[position - 1];
        position -= 1;
    }}
    ring_rows[position] = row;
    count += 1;
    row = {rows.parent("row")};
    guard += 1;
}}"""
    skip = f"""// Tangent row i belongs to a unit, which its anchor's visit updates.
const int parent = {rows.parent("i")};
if (parent < 0 || parent >= m) return false;
if ({rows.kind("parent")} != {CONTACT}) return false;
return HAS_PAIR(parent);"""
    anchor_units = """// A ring's members in row order: anchors as 3x3 units with the others' load held, the rest scalar.
for (int a = 0; a < normals; ++a) {
    const int row = ring_rows[a];
    if (HAS_PAIR(row)) {
        float ring_load = 0.0f;
        for (int q = 0; q < normals; ++q) {
            if (q != a) ring_load += lam[ring_rows[q]];
        }
        POINT_UNIT(row, ring_load, 1);
    } else {
        SCALAR_NORMAL(row);
    }
}"""
    lambdas = [
        (
            "UNIT_KEEP = [&](const int row, const int field, const float value)",
            "STATE(row, field) = value;",
        ),
        ("HAS_PAIR = [&](const int n) -> bool", has_pair),
        ("UNIT_SHAPE = [&](const int n, int& normals, int& anchors, int& first)", shape),
        ("SORT_RING = [&](const int first)", sort),
        ("SKIP_FRICTION = [&](const int i) -> bool", skip),
        ("CONTACT_WEIGHT = [&](const int r) -> float", f"return regularize != 0 ? {rows.weight('r')} : 1.0f;"),
        ("UNIT_RESIDUAL = [&](const int i) -> float", f"// Row i's whole residual.\n{residual}\nreturn {whole};"),
        ("UNIT_DIAGONAL = [&](const int r) -> float", sweep.unit_diagonal),
        ("UNIT_ENTRY = [&](const int r, const int c) -> float", sweep.unit_entry),
    ]
    if op.applies:
        apply = op.apply("delta", "").lstrip("\n")
        lambdas.append(("UNIT_APPLY = [&](const int i, const float delta)", f"{sweep.entries()}\n{apply}"))
    lambdas += [
        (
            "SCALAR_NORMAL = [&](const int i)",
            f"// A ring member without a tangent pair: the contact row's update.\n{scalar}",
        ),
        ("POINT_UNIT = [&](const int n, const float ring_load, const int anchor)", point),
        ("ANCHOR_UNITS = [&](const int normals)", anchor_units),
        ("RING_UNIT = [&](const int normals)", ring),
    ]
    state = ", ".join(f"{name} = {index}" for index, name in enumerate(UNIT_STATE))
    block = f"unit_block.data + world * {RING_BLOCK_SIZE}"
    storage = f"""    // Contact units: each row's unit state, a ring's rows and its block.
    const int {fields} = unit_state.shape[2];
    float* unit = unit_state.data + world * unit_state.shape[1] * {fields};
    const int {state};
    const auto STATE = [&](const int row, const int field) -> float& {{ return unit[row * {fields} + field]; }};
    int* ring_rows = unit_rows.data + world * unit_rows.shape[1];
    float* ring_W = {block} + {RING_BLOCK};
    float* ring_b = {block} + {RING_RESIDUAL};
    float* ring_old = {block} + {RING_IMPULSE};
    float* ring_x = {block} + {RING_SOLUTION};
    float* ring_alpha = {block} + {RING_ALPHA};
    // A launch at iteration offset 0 starts a solve: its slide roots and latches start clear.
    if (iteration_offset == 0) {{
        for (int r = 0; r < m * {fields}; ++r) unit[r] = 0.0f;
    }}"""
    declarations = "\n".join([storage, *(_lambda(head, body) for head, body in lambdas)])

    admission = f"""if (friction && {sweep.kind} == {FRICTION} && SKIP_FRICTION(i)) continue;
if (friction && {sweep.kind} == {CONTACT}) {{
    int normals = 0, anchors = 0, first = 0;
    UNIT_SHAPE(i, normals, anchors, first);
    if (normals > 1 || anchors > 0) {{
        if (i == first) {{
            if (normals == 1) {{
                POINT_UNIT(i, 0.0f, 0);
            }} else {{
                SORT_RING(i);
                if (ring_units != 0 && normals <= {CONTACT_UNIT_RING_NORMALS} && anchors <= {CONTACT_UNIT_RING_ANCHORS}) {{
                    RING_UNIT(normals);
                }} else {{
                    ANCHOR_UNITS(normals);
                }}
            }}
        }}
        continue;
    }}
}}"""
    return declarations, admission


# ---------------------------------------------------------------------------------------------------------------------
# The sweeps
# ---------------------------------------------------------------------------------------------------------------------


def _speed_rows(sweep: LoopSweep, pad: str, laws: RowLaws) -> str:
    """A free body's speed row (a velocity-limit row), in row order: the stateless clamp of its linear (axis
    ``e_0``) or angular (axis ``e_3``) speed along the velocity (:func:`.sweep.speed_row`); the row keeps its
    correction."""
    rows, op = sweep.rows, sweep.operator
    speed = speed_row_update(
        op,
        laws,
        load=rows.load("i"),
        store=rows.store("i", "speed_row[0]"),
        lanes=Lanes(1),
        rhs=rows.rhs("i"),
        inverse="mf_diag",
        pad="    ",
    )
    lines = [
        f"if ({sweep.kind} == {VELOCITY_LIMIT}) {{",
        f"    {sweep.declare}",
        f"    const int dof_a = BODY_DOF({rows.at('mf_body_a', 'i')});",
        "    if (mf_diag <= 0.0f || dof_a < 0) continue;",
        f"    const int speed_axis = {op.Z_a}.data[{op.base} + i * 6 + 3] != 0.0f ? 3 : 0;",
        speed,
        "    continue;",
        "}",
    ]
    return indent("\n".join(lines), pad)


def _source(sweep: LoopSweep, *, units: bool, speed_rows: bool, solve_policy: SolvePolicy) -> str:
    """The native body of a loop sweep: one thread sweeps a world's rows in order; the world stops under the solve
    policy (the thread's own vote is the world's)."""
    laws = _laws(solve_policy)
    pad = "            "
    ctx = sweep.context()
    row = row_visit(
        ctx,
        laws,
        parent=sweep.rows.parent("i"),
        check=sweep.check,
        declare=sweep.declare,
        entries=sweep.entries(),
        pad=pad,
    )
    declarations, admission = contact_units(sweep, laws) if units else ("", "")
    declarations = f"{declarations}\n" if declarations else ""
    visit = [
        f"{pad}const int {sweep.kind} = {sweep.rows.kind('i')};",
        indent(admission, pad) if admission else "",
        _speed_rows(sweep, pad, laws) if speed_rows else "",
        row,
    ]
    friction = "        const bool friction = global_iter >= friction_start_iteration;\n" if units else ""
    quiet = "iteration_changed == 0"
    sweep_end = sweep_exit(
        solve_policy,
        quiet=quiet,
        gate="global_iter >= friction_start_iteration",
        iteration="it",
        lane="0",
        argmax=None,
    )
    body = "\n".join(line for line in visit if line)
    return f"""    const int world = tid;
    const int m = {sweep.count}.data[world];
    if (m == 0) return;
    // Every per-row array of the world holds `capacity` rows.
    const int capacity = {sweep.impulses}.shape[1];
    const int {sweep.rows.offset} = world * capacity;
{indent(sweep.setup, "    ")}
    float* lam = {sweep.impulses}.data + {sweep.rows.offset};
    // The rows' change votes (the units' lambdas vote into it too).
    int iteration_changed = 0;
{declarations}    for (int it = 0; it < iterations; it++) {{
        const int global_iter = iteration_offset + it;
        iteration_changed = 0;
{friction}        for (int i = 0; i < m; i++) {{
{body}
        }}
{sweep_end}    }}
{indent(sweep.finish, "    ")}
"""


def loop_source(*, contact_units: bool = False, solve_policy: SolvePolicy = DEFAULT_POLICY) -> tuple[str, Staging]:
    """The native body of split mode's loop sweep over a world's assembled Delassus rows: one thread per world.

    The rows' impulses are updated in place; ``C``, the right-hand sides and the diagonals are read where they are
    used (:class:`.Assembled` with one thread: a row's residual from its right-hand side, ``C(i, j) lambda_j`` in
    column order). Contact, position-limit and velocity-limit rows are unilateral, a friction pair is solved at its
    first tangent on its patch ring's pooled load, every other row is bilateral; a contact's step takes its
    regularization weight when ``regularize`` is set. With ``contact_units`` a contact unit is solved exactly at its
    first row once friction is active (:func:`contact_units`). A world stops under the solve policy.

    Returns the body (its preludes: the row law's CPU math, the patch ring, the projection laws and, with units, the
    contact-unit law) and its staging (none: the sweep runs in place).
    """
    return _source(_dense(), units=contact_units, speed_rows=False, solve_policy=solve_policy), Staging()


def mf_loop_source(*, contact_units: bool = False, solve_policy: SolvePolicy = DEFAULT_POLICY) -> tuple[str, Staging]:
    """The native body of split mode's loop sweep over a world's free-body rows: one thread per world.

    The rows' impulses and the bodies' velocities are updated in place (:class:`.FreeBodyZ` with one thread in the
    loop's order, each body at its articulation's first coordinate, in its factor coordinates between the sweeps'
    entry and exit). Contact rows are unilateral; a friction pair is
    solved at its first tangent on its patch ring's pooled load, on the residuals and coupling it reduces itself (the
    loop's friction step); a velocity-limit row is its body's speed row; every other row is bilateral. A contact's
    step takes its regularization weight when ``regularize`` is set. With ``contact_units`` a contact unit is solved
    exactly at its first row once friction is active (:func:`contact_units`). A world stops under the solve policy.

    Returns the body and its staging (none: the sweep runs in place).
    """
    return _source(_free_body(), units=contact_units, speed_rows=True, solve_policy=solve_policy), Staging()


def _kernel(name: str, abi, body: str, preludes, solve_policy: SolvePolicy) -> wp.Kernel:
    return KernelTemplate(
        name + solve_policy.suffix,
        kernel_template.operands(abi, solve_policy.arguments),
        body,
        dispatch=kernel_template.THREADS,
        preludes=(*preludes, *solve_policy.preludes),
        dual=True,
        native_name=f"{name}_native",
    ).kernel()


@functools.cache
def loop_kernel(*, solve_policy: SolvePolicy = DEFAULT_POLICY) -> wp.Kernel:
    """The assembled loop sweep kernel (:func:`loop_source`); a readout's operand follows the rows' operands."""
    return _kernel("sweep_loop", LOOP_ABI, loop_source(solve_policy=solve_policy)[0], _PRELUDES, solve_policy)


@functools.cache
def loop_unit_kernel(*, solve_policy: SolvePolicy = DEFAULT_POLICY) -> wp.Kernel:
    """The assembled loop sweep kernel with contact units (:func:`loop_source`)."""
    source = loop_source(contact_units=True, solve_policy=solve_policy)[0]
    return _kernel("sweep_loop_unit", LOOP_UNIT_ABI, source, _UNIT_PRELUDES, solve_policy)


@functools.cache
def mf_loop_kernel(*, solve_policy: SolvePolicy = DEFAULT_POLICY) -> wp.Kernel:
    """The free-body loop sweep kernel (:func:`mf_loop_source`)."""
    return _kernel("sweep_mf_loop", MF_LOOP_ABI, mf_loop_source(solve_policy=solve_policy)[0], _PRELUDES, solve_policy)


@functools.cache
def mf_loop_unit_kernel(*, solve_policy: SolvePolicy = DEFAULT_POLICY) -> wp.Kernel:
    """The free-body loop sweep kernel with contact units (:func:`mf_loop_source`)."""
    source = mf_loop_source(contact_units=True, solve_policy=solve_policy)[0]
    return _kernel("sweep_mf_loop_unit", MF_LOOP_UNIT_ABI, source, _UNIT_PRELUDES, solve_policy)

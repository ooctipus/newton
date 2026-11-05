# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The native sparse PGS sweep as a configuration of the sweep emitter (:mod:`.sweep`).

The sweep solves a world's rows in factor coordinates: row ``i`` is ``Z_i = L^-1 P J_i^T``, the walk's footprints
packed as the coordinates and values of its support (:func:`.walk_rows.support_rows_kernel`), and the velocity it
acts on is the factor-coordinate delta from ``v_hat``, staged per world and published once
(``factor_velocity_delta``); the step's one leave walks it down (:func:`.factor.walk.walk_leave_kernel`, at
``Sweep.support_du_base`` and the class's nodes). A configuration chooses:

- the **mapping**: 32, 16 or 8 lanes per world, ``64 / lanes`` worlds per block of 64 threads, with launch bounds
  from the register line and the data (:data:`REGISTER_LINE`);
- the **operator's layout**: packed rows read through explicit row indices (``row_dof``) or the shared support
  table, or (``local_factors``) packed rows expanded once per call into dense rows in shared memory;
- the **schedule**: every row in slot order, or (``active_rows``) the rows selected at sweep entry, popped from
  a bitset (:func:`active_scan`); patch rings (``linked_rows``) select their friction pairs through the ring; under
  contact ``torsion`` the friction triples (a contact's pair and its spin row) in a phase of their own after each
  sweep's rows (:func:`triple_phase`), so the row loop carries none of their code;
- the **contact unit** (``contact_triplets``): a contact normal and its friction pair in one visit on their Gram
  block (:func:`contact_units`): one load group with the triplet's block record, one reduction of the three
  residuals, the normal step, the pair's residuals moved by the normal change through the record's couplings, the
  pair law from the record's metric, and one apply;
- the **laws**: the shared row visit and law (:func:`.sweep.row_visit`) for every other row, the last-word
  velocity-limit pass under either law (``velocity_limits``), the warm-start seed (``warm_start``), and the exit under
  the solve policy.

:func:`sparse_kernel` builds the kernel (:data:`SPARSE_ABI`, tiles of 64 threads) and :func:`sparse_staging` its
working set, which the solver's admission and the factory check against the device's static budget.
"""

from __future__ import annotations

import dataclasses
from functools import cache

import warp as wp

from ...core.execution import Work, shape
from ...core.memory import Buffer, Staging, arch_budget, built_for
from . import kernel_template
from .device_preludes import RING, SHARED_STORAGE, cuda_native, lane_group
from .kernel_template import KernelTemplate
from .law_source import law_preludes
from .projection_laws import CONE_LAW, PROJECTION_LAW
from .response_operator import Lanes, SparseFactor
from .row_inputs import RowArgument
from .solve_policy import DEFAULT_POLICY, Change, SolvePolicy, bound
from .sparse_contact import CONTACT_RECORD_FLOATS, CONTACT_RING_FLOATS
from .sparse_mass_matrix import _SparseRowSupport
from .sweep import (
    CONTACT,
    FRICTION,
    JOINT_LIMIT,
    TORSION,
    UNIT_HELD,
    VELOCITY_LIMIT,
    CachedFrictionMetric,
    RowContext,
    RowLaws,
    UnitOperator,
    cached_pair_law,
    contact_unit,
    end_sweep,
    friction_triple_visit,
    impulse_seed,
    indent,
    material_arguments,
    row_visit,
    spin_row,
    unit_held_loads,
    velocity_limit_pass,
    weight_arguments,
)

_SPARSE_LOCAL_WORKSPACE_BYTES = 8 * 1024
REGISTER_LINE = 64
"""The sparse sweeps' register line: 16 blocks of 64 threads a multiprocessor. Every sparse configuration compiles
within it without local memory. Its launch bound is the layer's (:func:`~newton._src.core.execution.shape`): at
most those blocks, as many as the working set's shared memory admits; ptxas fits the sweep's registers to that
occupancy, so a configuration whose shared memory allows 16 blocks keeps them however many registers its laws hold
live, and one that shared memory limits keeps its registers."""


SPARSE_ABI = kernel_template.params(
    ("world", int),
    ("world_constraint_count", wp.array[int]),
    ("dense_first", wp.array[int]),
    ("rhs_bias", wp.array2d[float]),
    ("world_diag", wp.array2d[float]),
    ("world_impulses", wp.array2d[float]),
    ("row_dof", wp.array3d[int]),
    ("row_factor", wp.array3d[float]),
    ("row_incident", wp.array2d[float]),
    ("world_row_type", wp.array2d[int]),
    ("world_row_parent", wp.array2d[int]),
    ("world_row_mu", wp.array2d[float]),
    ("world_dof_indices", wp.array2d[int]),
    ("v_hat", wp.array[float]),
    ("mf_constraint_count", wp.array[int]),
    ("mf_meta", wp.array2d[int]),
    ("mf_Z", wp.array3d[float]),
    ("iterations", int),
    ("omega", float),
    ("factor_velocity_delta", wp.array2d[float]),
    ("support", _SparseRowSupport),
)
"""The sparse sweep's native ABI (one world per call). ``world_dof_indices``, ``v_hat``, the ``mf_*`` operands and
the factory's free-body capacities are read by no configuration: the sweep holds no free-body rows."""

THREADS = 64
"""Threads per block: ``64 / lanes`` worlds."""


# ---------------------------------------------------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class SparseSweep:
    """A sparse sweep's configuration, as the factory's options resolve.

    Contact triplets need the support table, three rows, and the active-row scan or linked rows; linked rows
    take part only in an active-row scan or a contact unit. Under contact ``torsion`` a contact's spin row follows its
    pair, and the three are a friction triple, solved in the triple phase (:func:`triple_phase`).
    """

    rows: int
    dofs: int
    support: int
    lanes: int
    active: bool
    table: bool
    triplets: bool
    linked: bool
    velocity_limits: bool
    local: bool
    warm_start: bool
    torsion: bool
    policy: SolvePolicy
    weighted: bool = False
    material: bool = False

    @property
    def groups(self) -> int:
        """Worlds per block."""
        return THREADS // self.lanes

    @property
    def name(self) -> str:
        """The kernel's name."""
        suffix = "_active" if self.active else ""
        suffix += "_support_table" if self.table else ""
        suffix += "_contact_triplets" if self.triplets else ""
        suffix += "_linked" if self.linked else ""
        suffix += "_velocity_limits" if self.velocity_limits else ""
        suffix += f"_lanes{self.lanes}" if self.lanes != 32 else ""
        suffix += "_local_factors" if self.local else ""
        suffix += "_warm_start" if self.warm_start else ""
        suffix += "_weighted" if self.weighted else ""
        suffix += "_material" if self.material else ""
        suffix += "_torsion" if self.torsion else ""
        return f"sweep_sparse_{self.rows}_{self.dofs}_{self.support}{suffix}{self.policy.suffix}"


def _resolve(
    max_constraints: int,
    max_world_dofs: int,
    max_row_dofs: int,
    *,
    mf_max_constraints: int = 0,
    free_row_dofs: int = 0,
    active_rows: bool = False,
    use_support_table: bool = False,
    contact_triplets: bool = False,
    velocity_limits: bool = False,
    lanes_per_world: int = 32,
    local_factors: bool = False,
    warm_start: bool = False,
    linked_rows: bool = False,
    torsion: bool = False,
    solve_policy: SolvePolicy = DEFAULT_POLICY,
    weighted: bool = False,
    material: bool = False,
) -> SparseSweep:
    """Validate the factory's options and resolve the sweep they configure."""
    M, D, S = int(max_constraints), int(max_world_dofs), int(max_row_dofs)
    if int(mf_max_constraints) or int(free_row_dofs):
        raise ValueError("sparse PGS holds no physical free-body rows: mf_max_constraints and free_row_dofs must be 0")
    if M <= 0 or D <= 0 or S <= 0 or S > D:
        raise ValueError("sparse PGS requires positive capacities and max_row_dofs <= max_world_dofs")
    if use_support_table and D > 64:
        raise ValueError("sparse support tables require at most 64 factored DOFs")
    if velocity_limits and (active_rows or contact_triplets):
        raise ValueError("sparse joint velocity limits require full articulated row sweeps")
    L = int(lanes_per_world)
    if L not in (8, 16, 32):
        raise ValueError("sparse PGS requires 8, 16 or 32 lanes per world")
    if L < 32 and (D > L or active_rows or contact_triplets):
        raise ValueError("subwarp sparse PGS requires a small complete articulated sweep")
    if local_factors and (D > 16 or M > 64 or active_rows or contact_triplets):
        raise ValueError("local factor storage requires a small complete articulated sweep")
    active = bool(active_rows)
    triplets = bool(contact_triplets) and (active or bool(linked_rows)) and bool(use_support_table) and M >= 3 + torsion
    linked = bool(linked_rows) and (active or triplets)
    return SparseSweep(
        rows=M,
        dofs=D,
        support=S,
        lanes=L,
        active=active,
        table=bool(use_support_table),
        triplets=triplets,
        linked=linked,
        velocity_limits=bool(velocity_limits),
        local=bool(local_factors),
        warm_start=bool(warm_start),
        torsion=bool(torsion),
        policy=solve_policy,
        weighted=bool(weighted),
        material=bool(material),
    )


# ---------------------------------------------------------------------------------------------------------------------
# Row store and operator
# ---------------------------------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class SparseRows:
    """A world's rows in the sparse sweep.

    Impulses, right-hand sides (``rhs_bias + row_incident``), diagonals, their reciprocals and types are staged per
    world (``s_impulse``, ``s_rhs``, ``s_diag``, ``s_inverse``, ``s_type``); every lane reads them and the first lane
    stores the impulses. A row's diagonal is fixed for the solve: its reciprocal is formed once, at staging, and a
    row's step multiplies by it (the exit law reads the diagonal itself). Parents and friction coefficients stream from global memory; a patch ring walks at most ``count``
    rows. Under contact ``torsion`` a friction triple's pair (a contact with a spin row, :func:`.sweep.spin_row`) is
    staged as the spin row's kind: the row loop and the active scan skip the triple's three rows, and the triple
    phase finds its normal by the staged kinds and solves them (:func:`triple_phase`).
    """

    capacity: int
    lanes: Lanes
    torsion: bool = False

    def load(self, row: str) -> str:
        return f"s_impulse[{row}]"

    def store(self, row: str, value: str) -> str:
        return self.lanes.first(f"s_impulse[{row}] = {value};")

    def rhs_of(self, row: str) -> str:
        return f"s_rhs[{row}]"

    def diag(self, row: str) -> str:
        return f"s_diag[{row}]"

    def inverse(self, row: str) -> str:
        return f"s_inverse[{row}]"

    def kind(self, row: str, *, scan: bool = False) -> str:
        return f"s_type[{row}]"

    def parent(self, row: str) -> str:
        return f"world_row_parent.data[offset + {row}]"

    def mu(self, row: str) -> str:
        return f"world_row_mu.data[offset + {row}]"

    def weight(self, row: str) -> str:
        return f"world_row_w.data[offset + {row}]"

    def eps(self, row: str) -> str:
        return f"world_row_eps.data[offset + {row}]"

    def ring_load(self, parent: str, load: str | None = None) -> str:
        """The pooled normal load of the patch ring of normal row ``parent``, from ``load`` (its impulse)."""
        return f"RING_LOAD({load or self.load(parent)}, {parent}, count, ring_parent, ring_impulse)"

    def buffers(self, dofs: int) -> tuple[Buffer, ...]:
        rows = self.capacity
        return (
            Buffer("s_delta", "float", dofs),
            Buffer("s_impulse", "float", rows),
            Buffer("s_rhs", "float", rows),
            Buffer("s_diag", "float", rows),
            Buffer("s_inverse", "float", rows),
            Buffer("s_type", "unsigned char", rows),
        )

    def loads(self, pad: str) -> str:
        lane, width = self.lanes.lane, self.lanes.width
        kind = "static_cast<unsigned char>(world_row_type.data[offset + i])"
        triple = ""
        if self.torsion:
            kind = "static_cast<unsigned char>(kind)"
            spin = spin_row("parent", "count", lambda row: f"world_row_type.data[offset + {row}]")
            triple = (
                f"{pad}    int kind = world_row_type.data[offset + i];\n"
                f"{pad}    if (kind == {FRICTION}) {{\n"
                f"{pad}        const int parent = world_row_parent.data[offset + i];\n"
                f"{pad}        if ({spin} >= 0) kind = {TORSION};\n"
                f"{pad}    }}\n"
                f"{pad}    if (kind == {TORSION}) staged_triples = true;\n"
            )
        # Whether the world holds a friction triple (a spin row): the triple phase runs only where one does.
        present = f"\n{pad}const bool world_triples = ANY(staged_triples);\n" if self.torsion else ""
        return (
            (f"{pad}bool staged_triples = false;\n" if self.torsion else "")
            + f"{pad}for (int i = {lane}; i < count; i += {width}) {{\n"
            f"{pad}    s_impulse[i] = world_impulses.data[offset + i];\n"
            f"{pad}    s_rhs[i] = rhs_bias.data[offset + i] + row_incident.data[offset + i];\n"
            f"{pad}    const float diagonal = world_diag.data[offset + i];\n"
            f"{pad}    s_diag[i] = diagonal;\n"
            f"{pad}    s_inverse[i] = 1.0f / diagonal;\n"
            f"{triple}{pad}    s_type[i] = {kind};\n"
            f"{pad}}}{present}"
        )


def _class_coordinate(row_class: str, support: int, entry: str = "entry") -> str:
    return f"support.class_dof.data[{row_class} * {support} + {entry}]"


def factor_operator(sweep: SparseSweep) -> SparseFactor:
    """The sweep's operator: packed rows through the support table or explicit row indices, expanded once per
    call into dense rows with ``local_factors``."""
    S = sweep.support
    if sweep.table:

        def index(row, entry):
            return _class_coordinate(f"support.row_class.data[offset + {row}]", S, entry)

    else:

        def index(row, entry):
            return f"row_dof.data[row_base + {row} * {S} + {entry}]"

    def factor(row, entry):
        return f"row_factor.data[row_base + {row} * {S} + {entry}]"

    return SparseFactor(S, index, factor, Lanes(sweep.lanes), dense=sweep.dofs if sweep.local else 0)


# ---------------------------------------------------------------------------------------------------------------------
# Schedules
# ---------------------------------------------------------------------------------------------------------------------


def active_selection(sweep: SparseSweep, pad: str) -> str:
    """Whether the sweep selects rows at entry: a valid relaxation and, for point contacts, no linked normal
    (a world with patch rings keeps every row)."""
    finite = "isfinite(omega) && omega >= 0.0f"
    if sweep.linked:
        return f"{pad}const bool select_active = {finite};"
    lines = [
        "bool linked = false;",
        f"for (int row = lane; row < count; row += {sweep.lanes})",
        f"    if (s_type[row] == {CONTACT} && world_row_parent.data[offset + row] >= 0) linked = true;",
        f"const bool select_active = BALLOT(linked) == 0u && {finite};",
    ]
    return indent("\n".join(lines), pad)


def active_scan(sweep: SparseSweep, op: SparseFactor, pad: str) -> str:
    """Select the rows a sweep visits, one lane per row, into the bitsets ``s_normal_words`` and ``s_selected_words``.

    A contact normal or joint limit is selected when it holds impulse or its residual at sweep entry is
    negative (or not finite), every other non-friction row always. A first tangent is selected while its normal is
    selected or its pair holds impulse, a patch pair while any normal of its ring is. A row a sweep's earlier updates
    violate waits for the next sweep. Under contact torsion the rows of a friction triple, staged as spin rows
    (:class:`SparseRows`), are never selected: the triple phase solves them.
    """
    held = "s_impulse[parent + 2] != 0.0f"
    serial = dataclasses.replace(op, lanes=Lanes(1, "0"))
    # The delta is zero at the first sweep unless warm starting seeded it.
    dot = serial.accumulate("dot", "row", "")
    if not sweep.warm_start:
        dot = "if (iter != 0) {\n" + indent(dot, "    ") + "\n}"
    if sweep.linked:
        tangent = (
            "const auto normal_selected = [&](int r) { return ((s_normal_words[r / 32] >> (r & 31)) & 1u) != 0u; };\n"
            f"selected = row == parent + 1 && (s_impulse[row] != 0.0f || {held}\n"
            "    || (parent >= 0 && (normal_selected(parent)\n"
            "        || RING_ANY(parent, count, ring_parent, normal_selected))));"
        )
    else:
        tangent = (
            "selected = row == parent + 1 && (\n"
            "    ((s_normal_words[parent / 32] >> (parent & 31)) & 1u) != 0u\n"
            f"    || s_impulse[row] != 0.0f || {held});"
        )
    lane_zero = op.lanes.first
    select = f"!select_active || (kind != {CONTACT} && kind != {JOINT_LIMIT}) || s_impulse[row] != 0.0f"
    # A triple's rows are the triple phase's: never selected.
    rowed = f"kind != {FRICTION} && kind != {TORSION}" if sweep.torsion else f"kind != {FRICTION}"
    text = f"""for (int first = 0; first < count; first += 32) {{
    const int row = first + lane;
    bool selected = false;
    if (row < count) {{
        const int kind = s_type[row];
        if ({rowed}) {{
            selected = {select};
            if (!selected && s_diag[row] > 0.0f) {{
                float dot = 0.0f;
{indent(dot, "                ")}
                const float residual = dot + s_rhs[row];
                selected = residual < 0.0f || !isfinite(residual);
            }}
        }}
    }}
    const unsigned mask = BALLOT(selected);
    {lane_zero("s_normal_words[first / 32] = mask;")}
}}
SYNC();
for (int first = 0; first < count; first += 32) {{
    const int row = first + lane;
    bool selected = false;
    if (row < count) {{
        selected = !select_active || ((s_normal_words[first / 32] >> lane) & 1u) != 0u;
        if (select_active && s_type[row] == {FRICTION}) {{
            const int parent = world_row_parent.data[offset + row];
{indent(tangent, "            ")}
        }}
    }}
    const unsigned mask = BALLOT(selected);
    {lane_zero("s_selected_words[first / 32] = mask;")}
}}
SYNC();"""
    return indent(text, pad)


def active_violation(sweep: SparseSweep, op: SparseFactor) -> tuple[str, str, str, str]:
    """The exit law on the rows an active sweep leaves unvisited (:func:`.solve_policy.sweep_exit`): whether a visit
    of some zero-impulse unilateral row would make a loud change (its violation times ``omega``, as the row's own
    update would), and the largest such change with its row's kind."""
    serial = dataclasses.replace(op, lanes=Lanes(1, "0"))
    # A visit would change the row's own velocity by its excess, at a zero impulse: velocity units on both sides.
    visit = Change("excess", bound("s_rhs[row]", "fabsf(dot)", "0.0f", diagonal="1.0f"), "kind", "1.0f")
    code = f"""float sweep_violation = 0.0f;
int sweep_violation_kind = -1;
int sweep_violation_loud = 0;
for (int first = 0; first < count; first += 32) {{
    const int row = first + lane;
    const int kind = row < count ? static_cast<int>(s_type[row]) : -1;
    if ((kind == {CONTACT} || kind == {JOINT_LIMIT}) && s_impulse[row] == 0.0f && s_diag[row] > 0.0f) {{
        float dot = 0.0f;
{serial.accumulate("dot", "row", "        ")}
        const float residual = dot + s_rhs[row];
        const float excess = isfinite(residual) ? omega * fmaxf(-residual, 0.0f) : __int_as_float(0x7f800000);
        if (excess != 0.0f && ({visit.loud})) sweep_violation_loud = 1;
        if (excess > sweep_violation) {{
            sweep_violation = excess;
            sweep_violation_kind = kind;
        }}
    }}
}}
ARGMAX(sweep_violation, sweep_violation_kind);
const bool sweep_violation_any = BALLOT(sweep_violation_loud != 0) != 0u;"""
    return code, "sweep_violation_any", "sweep_violation", "sweep_violation_kind"


def warm_start_seed(op: SparseFactor, pad: str) -> str:
    """Give the delta the response of the impulses the row buffer carries, before the first sweep
    (:func:`.sweep.impulse_seed`)."""
    return impulse_seed(
        "count", "s_impulse[i]", "s_type[i]", op.apply_row("i", "seed", "        "), op.lanes.sync(), pad=pad
    )


def ordinary_rows(rows: SparseRows, op: SparseFactor, laws: RowLaws, *, skip_velocity_limits: bool, pad: str) -> str:
    """Row ``i``'s visit and update under the shared row law (:func:`.sweep.row_visit`).

    The friction gate, then the first tangent of a pair on its ring's disk; the residual reduces the pair's
    second-tangent term and coupling with it (:meth:`.SparseFactor.residual`). Velocity-limit rows wait for the
    last-word pass (``skip_velocity_limits``). Under contact torsion the rows staged as spin rows, a friction triple's
    (:class:`SparseRows`), are the triple phase's: the loop skips them and carries none of the triple's code. The
    first lane stores the impulses after every lane has read them.
    """
    lanes = rows.lanes
    op = dataclasses.replace(op, diagonal=lambda row: "denom" if row == "i" else rows.diag(row), inverse=rows.inverse)
    # The row's and its pair's impulses are read once, at admission, before the reduction.
    held = {"i": "row_impulse", "sib": "pair_impulse"}
    ctx = RowContext(
        row="i",
        kind="row_type",
        operator=op,
        load=lambda row: held.get(row) or rows.load(row),
        store=rows.store,
        rhs=rows.rhs_of,
        mu=rows.mu,
        weight=rows.weight,
        eps=rows.eps,
        parent="parent",
        ring_load=rows.ring_load,
        lanes=lanes,
        fenced=False,
    )
    head = [
        f"int row_type = {rows.kind('i')};",
        f"if (row_type == {VELOCITY_LIMIT}) continue;" if skip_velocity_limits else "",
        # A friction triple's rows are the triple phase's (contact torsion).
        f"if (row_type == {TORSION}) continue;" if laws.torsion else "",
    ]
    residual = op.residual("jv", pad, sibling="parent + 2", paired=f"row_type == {FRICTION}")
    visit = row_visit(
        ctx,
        dataclasses.replace(laws, torsion=False),
        parent=rows.parent("i"),
        check="denom",
        declare=f"float denom = {rows.diag('i')};",
        hold=f"const float row_impulse = {rows.load('i')};\nfloat pair_impulse = 0.0f;",
        keep_next="pair_impulse",
        residual=residual,
        pad=pad,
    )
    return indent("\n".join(line for line in head if line), pad) + "\n" + visit


# ---------------------------------------------------------------------------------------------------------------------
# Contact units: a contact normal and its friction pair on their Gram block, by the row law
# ---------------------------------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class _UnitFactor(UnitOperator, CachedFrictionMetric, SparseFactor):
    """A triplet's rows in a contact unit (:class:`.sweep.UnitOperator`): the unit's one reduction gathers the three
    rows' entries, its Gram couplings and the pair's metric are the triplet's block record, and the normal steps by the
    record's ``1/W_nn``. Every sparse row keeps its diagonal ``A_ii`` (``s_diag``), so the
    unit's rows vote in the velocity form, as the sparse rows' storage gives it (the storage question, option 1)."""

    def step(self, residual: str, row: str) -> str:
        return f"(r == 0 ? -{residual} * unit_inverse : -{residual} / {self.diagonal(row)})"


def _record(width: int | str, row: str) -> str:
    """The address of the contact record of the triplet that holds row ``row``, records ``width`` floats apart."""
    return f"contact_records.data + (static_cast<size_t>(line) * contact_records.shape[1] + {row} / 3) * {width}"


def contact_units(sweep: SparseSweep, rows: SparseRows, op: SparseFactor, laws: RowLaws, pad: str) -> tuple[str, str]:
    """The contact unit on the triplet's Gram block (:func:`.sweep.contact_unit`): its setup (linked: the rings'
    records), and its visit of row ``i``.

    A contact normal followed by its two tangents in the staged metadata loads in one group: the three rows'
    entries (every stored entry, then masked by its coordinate), the triplet's block record (``contact_records``:
    the Gram couplings ``W_nt1, W_nt2, W_t1t2``, ``1/W_nn`` and the pair's eigenvalues and eigenbasis) and its
    friction coefficient; one reduction gives the three rows' velocity terms. The rows then run the
    row law (:class:`_UnitFactor`): the normal steps with the record's ``1/W_nn``, the pair solves at the new load from
    the record's metric, its residuals moved by the normal change through ``W_nt`` (in exact arithmetic the sequential
    sweep); each coordinate takes the normal's, then the second tangent's, then the first tangent's change and is
    written back once. With ``linked`` a pass at entry stores each anchor's other ring members, at most three, behind
    its record in the order the pair update pools them (-1: a longer ring, which keeps the walk). A row the guards
    decline keeps the ordinary update.

    Under contact torsion a contact with a spin row is a friction triple, its pair staged as spin rows
    (:class:`SparseRows`): the guards decline it, its normal takes the ordinary update and the triple phase its pair
    and spin row (:func:`triple_phase`). With ``linked`` the entry pass stores a triple's ring members too, which its
    visit pools.
    """
    S = sweep.support
    width = CONTACT_RECORD_FLOATS + (CONTACT_RING_FLOATS if sweep.linked else 0)
    count = op.entries
    pair_rows = f"s_type[normal + 1] != {FRICTION} || s_type[normal + 2] != {FRICTION}"
    if laws.torsion:
        # A triple's pair is staged as spin rows (SparseRows).
        pair_rows = f"s_type[normal + 1] != s_type[normal + 2] || (s_type[normal + 1] != {FRICTION} && s_type[normal + 1] != {TORSION})"
    setup = ""
    if sweep.linked:
        setup = f"""// Each anchor's other ring members, at most three, in the order the pair update pools them; -1 marks a
// longer ring, which keeps the walk. Visits read them with the record.
for (int normal = lane; normal + 2 < count; normal += 32) {{
    if (s_type[normal] != {CONTACT} || {pair_rows}) continue;
    const int first = world_row_parent.data[offset + normal];
    int second = -1, third = -1, others = 0;
    if (first >= 0 && first != normal) {{
        others = 1;
        second = world_row_parent.data[offset + first];
        if (second >= 0 && second != normal) {{
            others = 2;
            third = world_row_parent.data[offset + second];
            if (third >= 0 && third != normal) {{
                const int fourth = world_row_parent.data[offset + third];
                others = fourth >= 0 && fourth != normal ? -1 : 3;
            }}
        }}
    }}
    *reinterpret_cast<int4*>({_record(width, "normal")} + {CONTACT_RECORD_FLOATS})
        = make_int4(first, second, third, others);
}}
SYNC();"""
    entry_guard = "select_active && " if sweep.active else ""
    selected = "\n        & (((s_selected_words[(i + 1) / 32] >> ((i + 1) & 31)) & 1u) != 0u)" if sweep.active else ""
    normal_parent = "" if sweep.linked else "\n    const int normal_parent = world_row_parent.data[offset + i];"
    point_guard = "" if sweep.linked else " && normal_parent < 0"
    ring_read = (
        f"\n        const int4 ring = *reinterpret_cast<const int4*>(record + {CONTACT_RECORD_FLOATS});"
        if sweep.linked
        else ""
    )

    def ring_load(parent: str, load: str) -> str:
        # A point contact's pool is its normal's load.
        return load

    ring = ""
    if sweep.linked:
        # The ring pools as the pair update walks it, from the normal's new load on.
        ring = f"""
        const auto unit_ring_load = [&](float load) -> float {{
            if (ring.w < 0) return {rows.ring_load("head", "load")};
            if (ring.w > 0) load += s_impulse[ring.x];
            if (ring.w > 1) load += s_impulse[ring.y];
            if (ring.w > 2) load += s_impulse[ring.z];
            return load;
        }};"""

        def ring_load(parent: str, load: str) -> str:
            return f"unit_ring_load({load})"

    def body() -> str:
        """The unit's rows, loads, reduction and apply."""
        unit_rows = [
            ("i", "normal", "unit_jv0"),
            ("tangent", "tangent", "unit_jv1"),
            ("sibling", "sibling", "unit_jv2"),
        ]
        gather = op.entries_loop(
            "\n".join(
                [
                    f"const bool stored = entry < {S};",
                    *(f"const float {name}_entry = stored ? {op.element(row)} : 0.0f;" for row, name, _ in unit_rows),
                    f"const int index = stored ? {_class_coordinate('row_class', S)} : -1;",
                    "indices[q] = index;",
                    *(f"{name}_responses[q] = index >= 0 ? {name}_entry : 0.0f;" for _, name, _ in unit_rows),
                    "velocities[q] = index >= 0 ? s_delta[index] : 0.0f;",
                    "if (index >= 0) {",
                    *(f"    {term} += {name}_responses[q] * velocities[q];" for _, name, term in unit_rows),
                    "}",
                ]
            ),
            "        ",
        )
        # Per coordinate the normal's change, then the pair's: the row loop's order.
        changes = [("normal", "unit_dn"), ("sibling", "unit_dt2")]
        apply = op.entries_loop(
            "\n".join(
                [
                    "if (indices[q] >= 0) {",
                    "    float value = velocities[q];",
                    *(f"    if ({d} != 0.0f) value += {name}_responses[q] * {d};" for name, d in changes),
                    "    if (unit_dt1 != 0.0f) value += tangent_responses[q] * unit_dt1;",
                    "    s_delta[indices[q]] = value;",
                    "}",
                ]
            ),
            "            ",
            entry=False,
        )
        unit_op = _UnitFactor(**{field.name: getattr(op, field.name) for field in dataclasses.fields(op)})
        ctx = RowContext(
            row="i",
            kind="row_type",
            operator=dataclasses.replace(unit_op, diagonal=rows.diag),
            load=rows.load,
            store=rows.store,
            rhs=rows.rhs_of,
            mu=lambda row: "mu",
            weight=rows.weight,
            eps=rows.eps,
            parent="parent",
            ring_load=ring_load,
            lanes=rows.lanes,
            fenced=False,
            pair_law="UNIT_PAIR(0, ",
        )
        declare = f"""const int i = head + r;
const int row_type = r == 0 ? {CONTACT} : {FRICTION};"""
        # The guards admitted the normal (a positive diagonal); the rows read the unit's staged values as it loaded
        # them.
        unit_laws = dataclasses.replace(laws, torsion=False)
        rows_text = contact_unit(
            ctx,
            unit_laws,
            rows="3",
            declare=declare,
            parent="head",
            check=None,
            held=True,
            pad="        ",
        )
        held = "\n".join(unit_held_loads(ctx, name, "head", "        ") for name in UNIT_HELD)
        # The record's second half holds the pair's metric: its eigenvalues and the larger's axis.
        tail = "\n        const float4 basis = *reinterpret_cast<const float4*>(record + 4);"
        metric = cached_pair_law("UNIT_PAIR", "basis.x", "basis.y", ("basis.z", "basis.w"), "        ")
        return f"""        const float* record = {_record(width, "i")};
        const float4 coupling = *reinterpret_cast<const float4*>(record);{tail}{ring_read}
        const float mu = world_row_mu.data[offset + tangent];
        int indices[{count}];
        float velocities[{count}], normal_responses[{count}];
        float tangent_responses[{count}], sibling_responses[{count}];
        float unit_jv0 = 0.0f, unit_jv1 = 0.0f, unit_jv2 = 0.0f;
{gather}
        // The unit's staged values load with its rows, off the reduction's chain; the rows' law reads them.
{held}
        SUM(unit_jv0, unit_jv1, unit_jv2);
        // W_nt moves the tangent residuals by the normal change; the record's 1/W_nn steps the normal.
        const float gram_n1 = coupling.x, gram_n2 = coupling.y, unit_inverse = coupling.w;
{metric}{ring}
{rows_text}
        if (unit_dn != 0.0f || unit_dt1 != 0.0f || unit_dt2 != 0.0f) {{
{apply}
        }}"""

    units = body()
    update = f"""if (i == consumed_tangent) continue;
// The staged metadata first, read together: a contact normal with its two tangents.
if ({entry_guard}i + 2 < count
    && ((s_type[i] == {CONTACT}) & (s_type[i + 1] == {FRICTION}) & (s_type[i + 2] == {FRICTION})
        & (isfinite(s_diag[i]) != 0) & (s_diag[i] > 0.0f){selected})) {{
    const int head = i, tangent = i + 1, sibling = i + 2;
    // The guards load together; the rows, the record and the coordinates then load together.
    const int row_class = support.row_class.data[offset + i];
    const int tangent_class = support.row_class.data[offset + tangent];
    const int sibling_class = support.row_class.data[offset + sibling];
    const int tangent_parent = world_row_parent.data[offset + tangent];
    const int sibling_parent = world_row_parent.data[offset + sibling];{normal_parent}
    if (tangent_parent == i && sibling_parent == i && tangent_class == sibling_class
        && row_class == tangent_class{point_guard}) {{
{units}
        consumed_tangent = tangent;
        SYNC();
        continue;
    }}
}}"""
    return indent(setup, "    ") if setup else "", indent(update, pad)


# ---------------------------------------------------------------------------------------------------------------------
# Friction triples: a contact's pair and its spin row, in a phase of their own after each sweep's rows
# ---------------------------------------------------------------------------------------------------------------------


def triple_phase(sweep: SparseSweep, rows: SparseRows, op: SparseFactor, laws: RowLaws, pad: str) -> str:
    """The friction triples after a sweep's rows (contact torsion), in row order, each visited by
    :func:`.sweep.friction_triple_visit`: one reduction and one apply of its pair and spin row together, on the pooled
    normal load of its contact's patch ring. A triple's normal is a contact whose next row is staged as a spin row (its
    pair, :class:`SparseRows`): the lanes find them together, a lane a row.

    A triple's metric is its contact's record's (three eigenvalues and the eigenbasis as a quaternion), formed
    once per solve by the writer (:func:`.walk_rows.support_rows_kernel`), and loads with the ring ahead of the
    reduction. A linked sweep's contact units store each normal's ring members behind its record at entry
    (:func:`contact_units`), a triple's too: the visit pools them from the record, and a longer ring (or a sweep
    without them) walks the ring.
    """
    record = _record("contact_records.shape[2]", "head")
    ring = rows.ring_load("head", "pooled")
    metric = [
        f"const float* const record = {record};",
        "const float4 triple_metric = *reinterpret_cast<const float4*>(record);",
        "const float4 triple_basis = *reinterpret_cast<const float4*>(record + 4);",
    ]
    pool = [f"float pooled = {rows.load('head')};"]
    if sweep.linked and sweep.triplets:
        metric.append(f"const int4 ring = *reinterpret_cast<const int4*>(record + {CONTACT_RECORD_FLOATS});")
        pool += [
            f"if (ring.w < 0) pooled = {ring};",
            "if (ring.w > 0) pooled += s_impulse[ring.x];",
            "if (ring.w > 1) pooled += s_impulse[ring.y];",
            "if (ring.w > 2) pooled += s_impulse[ring.z];",
        ]
    else:
        pool.append(f"pooled = {ring};")
    law = (
        "ConeLaw::friction_triple_from_metric(wp::vec3(triple_metric.x, triple_metric.y, triple_metric.z),\n"
        "    wp::quat_to_matrix(wp::quat(triple_basis.x, triple_basis.y, triple_basis.z, triple_basis.w)), "
    )
    ctx = RowContext(
        row="tangent",
        kind=str(FRICTION),
        operator=dataclasses.replace(op, diagonal=rows.diag),
        load=rows.load,
        store=rows.store,
        rhs=rows.rhs_of,
        mu=rows.mu,
        weight=rows.weight,
        parent="head",
        ring_load=rows.ring_load,
        lanes=rows.lanes,
        fenced=False,
    )
    visit = friction_triple_visit(
        ctx, laws, head="head", metric="\n".join(metric), pool="\n".join(pool), load="pooled", law=law, pad="        "
    )
    # Only a world whose staging found a triple (world_triples, uniform over its lanes) scans for them.
    text = f"""if (world_triples) for (int first = 0; first < count; first += {sweep.lanes}) {{
    const int row = first + lane;
    unsigned heads = BALLOT(row + 1 < count && s_type[row] == {CONTACT} && s_type[row + 1] == {TORSION});
    while (heads != 0u) {{
        const int head = first + FIRST(heads);
        heads &= heads - 1u;
{visit}
    }}
}}"""
    return indent(text, pad)


# ---------------------------------------------------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------------------------------------------------


def _staging(sweep: SparseSweep, rows: SparseRows, op: SparseFactor) -> Staging:
    """Each world's shared working set: its rows and delta, the active rows' bitsets and the dense rows of
    ``local_factors``, one array per buffer for the block's worlds."""
    buffers = list(rows.buffers(sweep.dofs))
    if sweep.active:
        words = (sweep.rows + 31) // 32
        buffers += [Buffer("s_normal_words", "unsigned", words), Buffer("s_selected_words", "unsigned", words)]
    if op.dense:
        buffers.append(Buffer(op.local, "float", sweep.rows * op.dense))
    return Staging(tuple(buffers), groups=sweep.groups, threads=THREADS, layout="arrays")


def sparse_staging(max_constraints: int, max_world_dofs: int, max_row_dofs: int, **options) -> Staging:
    """The shared working set of the sweep :func:`sparse_kernel` builds from the same arguments, without its body."""
    sweep = _resolve(max_constraints, max_world_dofs, max_row_dofs, **options)
    return _staging(sweep, SparseRows(sweep.rows, Lanes(sweep.lanes), sweep.torsion), factor_operator(sweep))


def _source(sweep: SparseSweep) -> tuple[str, Staging]:
    """The sweep's native body: staging and setup, then per sweep the active scan, the row loop (contact unit, then
    the ordinary update), the friction triples (contact torsion), the last-word velocity-limit pass and the exit;
    then the publication."""
    M, D, S, L = sweep.rows, sweep.dofs, sweep.support, sweep.lanes
    lanes = Lanes(L)
    rows = SparseRows(M, lanes, sweep.torsion)
    op = factor_operator(sweep)
    laws = RowLaws(regularize=sweep.weighted, material=sweep.material, policy=sweep.policy, torsion=sweep.torsion)
    staging = _staging(sweep, rows, op)

    setup = []
    if op.dense:
        setup.append(op.expand("count", "    "))
    setup.append("    SYNC();")
    if sweep.active:
        setup.append(active_selection(sweep, "    "))
    if sweep.warm_start:
        setup.append(warm_start_seed(op, "    "))
    body_pad = "                " if sweep.active else "            "
    unit = ""
    if sweep.triplets:
        unit_setup, unit = contact_units(sweep, rows, op, laws, body_pad)
        if unit_setup:
            setup.append(unit_setup)

    scan = []
    if sweep.active:
        scan.append(active_scan(sweep, op, "        "))
    if sweep.triplets:
        scan.append("        int consumed_tangent = -1;")
    update = ordinary_rows(rows, op, laws, skip_velocity_limits=sweep.velocity_limits, pad=body_pad)
    if sweep.active:
        row_loop = f"""        for (int word = 0; word < (count + 31) / 32; ++word) {{
            unsigned pending = s_selected_words[word];
            while (pending != 0u) {{
                const int i = word * 32 + FIRST(pending);
                pending &= pending - 1u;
{unit}
{update}
            }}
        }}"""
    else:
        row_loop = f"""        for (int i = 0; i < count; ++i) {{
{unit}
{update}
        }}"""
    triples = triple_phase(sweep, rows, op, laws, "        ") + "\n" if sweep.torsion else ""
    passes = ""
    if sweep.velocity_limits:
        passes = velocity_limit_pass(rows, op, laws, end="count", pad="        ")
    exit_ = end_sweep(sweep.policy, iteration="iter", violation=active_violation(sweep, op) if sweep.active else None)
    body = f"""    const int lane = LANE;
    const int line = dense_first.data[world];
    __builtin_assume(line >= 0);
    const int offset = line * {M};
    const int row_base = offset * {S};
    const int count = min(world_constraint_count.data[world], {M});
{staging.declarations("    ")}
    // A contact normal's patch ring links the region's next normal (-1 ends a point contact).
    const auto ring_parent = [&](int r) {{ return world_row_parent.data[offset + r]; }};
    const auto ring_impulse = [&](int r) {{ return s_impulse[r]; }};

    for (int d = lane; d < {D}; d += {L}) s_delta[d] = 0.0f;
{rows.loads("    ")}
{chr(10).join(setup)}

    for (int iter = 0; iter < iterations; ++iter) {{
        int iteration_changed = 0;
{chr(10).join(scan)}
{row_loop}
{triples}{passes}
{exit_}    }}

    for (int d = lane; d < {D}; d += {L}) factor_velocity_delta.data[world * {D} + d] = s_delta[d];
    for (int i = lane; i < count; i += {L}) world_impulses.data[offset + i] = s_impulse[i];
#endif
"""
    cone = (CONE_LAW,) if sweep.torsion else ()
    preludes = (lane_group(L), SHARED_STORAGE, RING, *law_preludes(PROJECTION_LAW, *cone), *sweep.policy.preludes)
    return cuda_native(body, preludes=preludes), staging


def sparse_source(
    max_constraints: int,
    max_world_dofs: int,
    max_row_dofs: int,
    *,
    mf_max_constraints: int = 0,
    free_row_dofs: int = 0,
    active_rows: bool = False,
    use_support_table: bool = False,
    contact_triplets: bool = False,
    velocity_limits: bool = False,
    lanes_per_world: int = 32,
    local_factors: bool = False,
    warm_start: bool = False,
    linked_rows: bool = False,
    torsion: bool = False,
    solve_policy: SolvePolicy = DEFAULT_POLICY,
    weighted: bool = False,
    material: bool = False,
) -> tuple[str, Staging]:
    """The native body of the sparse sweep and its staging."""
    return _source(
        _resolve(
            max_constraints,
            max_world_dofs,
            max_row_dofs,
            mf_max_constraints=mf_max_constraints,
            free_row_dofs=free_row_dofs,
            active_rows=active_rows,
            use_support_table=use_support_table,
            contact_triplets=contact_triplets,
            velocity_limits=velocity_limits,
            lanes_per_world=lanes_per_world,
            local_factors=local_factors,
            warm_start=warm_start,
            linked_rows=linked_rows,
            torsion=torsion,
            solve_policy=solve_policy,
            weighted=weighted,
            material=material,
        )
    )


@cache
def sparse_kernel(
    max_constraints: int,
    max_world_dofs: int,
    max_row_dofs: int,
    *,
    mf_max_constraints: int = 0,
    free_row_dofs: int = 0,
    active_rows: bool = False,
    use_support_table: bool = False,
    contact_triplets: bool = False,
    velocity_limits: bool = False,
    lanes_per_world: int = 32,
    local_factors: bool = False,
    warm_start: bool = False,
    linked_rows: bool = False,
    torsion: bool = False,
    solve_policy: SolvePolicy = DEFAULT_POLICY,
    device_arch=None,
    weighted: bool = False,
    material: bool = False,
) -> wp.Kernel:
    """The sparse sweep kernel.

    Launch ``ceil(world_count / (64 / lanes_per_world))`` tiles of 64 threads. Contact units take
    ``contact_records`` after ``omega``; a solve readout appends its operand after ``support``. The
    working set must fit the static budget of ``device_arch`` (:func:`.memory.arch_budget`).
    """
    sweep = _resolve(
        max_constraints,
        max_world_dofs,
        max_row_dofs,
        mf_max_constraints=mf_max_constraints,
        free_row_dofs=free_row_dofs,
        active_rows=active_rows,
        use_support_table=use_support_table,
        contact_triplets=contact_triplets,
        velocity_limits=velocity_limits,
        lanes_per_world=lanes_per_world,
        local_factors=local_factors,
        warm_start=warm_start,
        linked_rows=linked_rows,
        torsion=torsion,
        solve_policy=solve_policy,
        weighted=weighted,
        material=material,
    )
    snippet, staging = _source(sweep)
    staging.check("the sparse sweep", arch_budget(device_arch))
    extra = (*weight_arguments(sweep.weighted), *material_arguments(sweep.material), *sweep.policy.arguments)
    if sweep.triplets or sweep.torsion:
        # Contact units read their records; where spin rows may appear every triple reads its metric there.
        extra = (RowArgument("contact_records", wp.array3d[float], after="omega"), *extra)
    return KernelTemplate.from_abi(
        sweep.name,
        SPARSE_ABI,
        snippet,
        kernel_template.items("world", groups=sweep.groups, lanes=sweep.lanes, guard="world_count"),
        extra=extra,
        composed=True,
        native_name="sweep_sparse_native",
        launch_bounds=shape(
            Work(0, block=THREADS, registers=REGISTER_LINE, shared=staging.bytes), built_for(device_arch)
        ).bounds,
    ).kernel()

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

:class:`SparseSweep` is the configuration the one factory builds (:func:`.sweep.sweep_kernel`: :data:`SPARSE_ABI`,
tiles of 64 threads); its working set (:attr:`SparseSweep.staging`) is what the solver's admission and the factory
check against the device's static budget.
"""

from __future__ import annotations

import dataclasses
from typing import ClassVar

import warp as wp

from ...core.execution import Work, device_lanes
from ...core.memory import Buffer, Staging
from . import kernel_template
from .kernel_template import Dispatch
from .response_operator import Lanes, SparseFactor
from .row_inputs import RowArgument
from .solve_policy import Change, bound
from .sparse_contact import CONTACT_RECORD_FLOATS, CONTACT_RING_FLOATS
from .sparse_mass_matrix import _SparseRowSupport
from .sweep import (
    CONTACT,
    FRICTION,
    ITERATION,
    JOINT_LIMIT,
    TORSION,
    UNIT_HELD,
    VELOCITY_LIMIT,
    BallotHeads,
    Diagonal,
    RowContext,
    RowLaws,
    SelectedRows,
    SlotOrder,
    StagedRows,
    SweepConfig,
    UnitZ,
    cached_pair_law,
    contact_unit,
    end_sweep,
    friction_triple_visit,
    impulse_seed,
    indent,
    level_loop,
    row_step,
    unit_held_loads,
    velocity_limit_pass,
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
    ("iterations", int),
    ("omega", float),
    ("factor_velocity_delta", wp.array2d[float]),
    ("support", _SparseRowSupport),
)
"""The sparse sweep's native ABI (one world per call). ``world_dof_indices`` and ``v_hat`` are read by no
configuration: the sweep holds no free-body rows."""

THREADS = 64
"""Threads per block: ``64 / lanes`` worlds."""


# ---------------------------------------------------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class SparseSweep(SweepConfig):
    """The native sparse sweep's configuration (case K1): a world's rows (at most ``max_constraints``) in factor
    coordinates of its one tree of ``max_world_dofs`` coordinates, each row's support at most ``max_row_dofs`` entries
    (:func:`_source`).

    Its switches, each the plan's from the class's shape and the laws (the class plan's packed choice,
    :meth:`.sweep_plan.Sweep.place_packed`):

    - the support class: ``use_support_table``, the rows' coordinates from the shared support table (at most 64
      factored coordinates), else explicit row indices; ``local_factors`` expands the packed rows into dense rows in
      shared memory once a call;
    - lanes: ``lanes_per_world`` (8, 16 or 32) a world, ``64 / lanes`` worlds a block (the device's: one thread on the
      CPU, :attr:`lanes`);
    - the enumeration: ``active_rows`` visits the rows selected at sweep entry (:func:`active_scan`);
    - the contact unit: ``contact_triplets``, a contact normal and its friction pair on their Gram block, with the
      support table and the active scan or linked rows, three rows (four with torsion) at least; ``linked_rows``
      (patch rings) take part only in an active scan or a contact unit;
    - ``velocity_limits``: the velocity-limit rows' last word; ``warm_start``: the seed.

    The configuration resolves its request: a contact unit or linked rows it cannot hold are off.
    """

    max_constraints: int
    max_world_dofs: int
    max_row_dofs: int
    _: dataclasses.KW_ONLY
    active_rows: bool = False
    use_support_table: bool = False
    contact_triplets: bool = False
    velocity_limits: bool = False
    lanes_per_world: int = 32
    local_factors: bool = False
    warm_start: bool = False
    linked_rows: bool = False
    device_arch: str | None = None

    ABI: ClassVar[tuple] = SPARSE_ABI
    NATIVE: ClassVar[str] = "sweep_sparse_native"
    LABEL: ClassVar[str] = "the sparse sweep"

    def __post_init__(self):
        M, D, S = int(self.max_constraints), int(self.max_world_dofs), int(self.max_row_dofs)
        if M <= 0 or D <= 0 or S <= 0 or S > D:
            raise ValueError("sparse PGS requires positive capacities and max_row_dofs <= max_world_dofs")
        if self.use_support_table and D > 64:
            raise ValueError("sparse support tables require at most 64 factored DOFs")
        if int(self.lanes_per_world) not in (8, 16, 32):
            raise ValueError("sparse PGS requires 8, 16 or 32 lanes per world")
        active = bool(self.active_rows)
        triplets = (
            bool(self.contact_triplets)
            and (active or bool(self.linked_rows))
            and bool(self.use_support_table)
            and M >= 3 + self.torsion
        )
        resolved = {
            "max_constraints": M,
            "max_world_dofs": D,
            "max_row_dofs": S,
            "lanes_per_world": int(self.lanes_per_world),
            "active_rows": active,
            "use_support_table": bool(self.use_support_table),
            "contact_triplets": triplets,
            "linked_rows": bool(self.linked_rows) and (active or triplets),
            "velocity_limits": bool(self.velocity_limits),
            "local_factors": bool(self.local_factors),
            "warm_start": bool(self.warm_start),
            "torsion": bool(self.torsion),
            "weighted": bool(self.weighted),
            "material": bool(self.material),
        }
        for field, value in resolved.items():
            object.__setattr__(self, field, value)

    @property
    def lanes(self) -> int:
        """A world's lanes on the device (:func:`.execution.device_lanes`: one on the CPU)."""
        return device_lanes(self.device_arch, self.lanes_per_world)

    @property
    def threads(self) -> int:
        """Threads a block: :data:`THREADS` on CUDA, one on the CPU (:func:`.execution.device_lanes`)."""
        return device_lanes(self.device_arch, THREADS)

    @property
    def groups(self) -> int:
        """Worlds per block."""
        return self.threads // self.lanes

    @property
    def name(self) -> str:
        suffix = "_active" if self.active_rows else ""
        suffix += "_support_table" if self.use_support_table else ""
        suffix += "_contact_triplets" if self.contact_triplets else ""
        suffix += "_linked" if self.linked_rows else ""
        suffix += "_velocity_limits" if self.velocity_limits else ""
        suffix += f"_lanes{self.lanes}" if self.lanes != 32 else ""
        suffix += "_local_factors" if self.local_factors else ""
        suffix += "_warm_start" if self.warm_start else ""
        suffix += "_weighted" if self.weighted else ""
        suffix += "_material" if self.material else ""
        suffix += "_torsion" if self.torsion else ""
        rows, dofs, support = self.max_constraints, self.max_world_dofs, self.max_row_dofs
        return f"sweep_sparse_{rows}_{dofs}_{support}{suffix}{self.solve_policy.suffix}"

    @property
    def staging(self) -> Staging:
        """The sweep's shared working set, without its body (:func:`_staging`)."""
        return _staging(
            self, StagedRows(self.max_constraints, Lanes(self.lanes), triples=self.torsion), factor_operator(self)
        )

    def source(self) -> tuple[str, Staging]:
        return _source(self)

    def dispatch(self, staging: Staging) -> Dispatch:
        return kernel_template.items("world", groups=self.groups, lanes=self.lanes, guard="world_count")

    def operands(self) -> tuple[RowArgument, ...]:
        # Contact units read their records; where spin rows may appear every triple reads its metric there.
        records = self.contact_triplets or self.torsion
        return (
            *((RowArgument("contact_records", wp.array3d[float], after="omega"),) if records else ()),
            *self.law_operands(),
        )

    def work(self, items: int = 0, shared: int = 0) -> Work:
        """``lanes_per_world`` lanes a world in blocks of :data:`THREADS` (one thread a world on the CPU), at the
        register line :data:`REGISTER_LINE`."""
        return Work(int(items), lanes=self.lanes_per_world, block=THREADS, registers=REGISTER_LINE, shared=int(shared))


# ---------------------------------------------------------------------------------------------------------------------
# Row store and operator
# ---------------------------------------------------------------------------------------------------------------------


def _class_coordinate(row_class: str, support: int, entry: str = "entry") -> str:
    return f"support.class_dof.data[{row_class} * {support} + {entry}]"


def factor_operator(sweep: SparseSweep) -> SparseFactor:
    """The sweep's operator: packed rows through the support table or explicit row indices, expanded once per
    call into dense rows with ``local_factors``."""
    S = sweep.max_row_dofs
    if sweep.use_support_table:

        def index(row, entry):
            return _class_coordinate(f"support.row_class.data[offset + {row}]", S, entry)

    else:

        def index(row, entry):
            return f"row_dof.data[row_base + {row} * {S} + {entry}]"

    def factor(row, entry):
        return f"row_factor.data[row_base + {row} * {S} + {entry}]"

    return SparseFactor(S, index, factor, Lanes(sweep.lanes), dense=sweep.max_world_dofs if sweep.local_factors else 0)


# ---------------------------------------------------------------------------------------------------------------------
# Schedules
# ---------------------------------------------------------------------------------------------------------------------


def active_selection(sweep: SparseSweep, pad: str) -> str:
    """Whether the sweep selects rows at entry: a valid relaxation and, for point contacts, no linked normal
    (a world with patch rings keeps every row)."""
    finite = "isfinite(omega) && omega >= 0.0f"
    if sweep.linked_rows:
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
    (:class:`.sweep.StagedRows`), are never selected: the triple phase solves them.
    """
    held = "s_impulse[parent + 2] != 0.0f"
    serial = dataclasses.replace(op, lanes=Lanes(1, "0"))
    # The delta is zero at the first sweep unless warm starting seeded it.
    dot = serial.dot((("dot", "row", None),), {"row": "row"}, "", reduce=False)
    if not sweep.warm_start:
        dot = "if (iter != 0) {\n" + indent(dot, "    ") + "\n}"
    if sweep.linked_rows:
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
    lane_zero, L = op.lanes.first, sweep.lanes
    # A word holds 32 rows' bits; a group of fewer lanes fills it L bits at a time, from its first lane.
    if L == 32:
        bit, fill = "lane", lambda words: f"{words}[first / 32] = mask;"
    else:
        bit = "((first & 31) + lane)"

        def fill(words: str) -> str:
            kept = f"((first & 31) != 0 ? {words}[first / 32] : 0u)"
            return f"{words}[first / 32] = {kept} | (mask << (first & 31));"

    select = f"!select_active || (kind != {CONTACT} && kind != {JOINT_LIMIT}) || s_impulse[row] != 0.0f"
    # A triple's rows are the triple phase's: never selected.
    rowed = f"kind != {FRICTION} && kind != {TORSION}" if sweep.torsion else f"kind != {FRICTION}"
    text = f"""for (int first = 0; first < count; first += {L}) {{
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
    {lane_zero(fill("s_normal_words"))}
}}
SYNC();
for (int first = 0; first < count; first += {L}) {{
    const int row = first + lane;
    bool selected = false;
    if (row < count) {{
        selected = !select_active || ((s_normal_words[first / 32] >> {bit}) & 1u) != 0u;
        if (select_active && s_type[row] == {FRICTION}) {{
            const int parent = world_row_parent.data[offset + row];
{indent(tangent, "            ")}
        }}
    }}
    const unsigned mask = BALLOT(selected);
    {lane_zero(fill("s_selected_words"))}
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
for (int first = 0; first < count; first += {sweep.lanes}) {{
    const int row = first + lane;
    const int kind = row < count ? static_cast<int>(s_type[row]) : -1;
    if ((kind == {CONTACT} || kind == {JOINT_LIMIT}) && s_impulse[row] == 0.0f && s_diag[row] > 0.0f) {{
        float dot = 0.0f;
{serial.dot((("dot", "row", None),), {"row": "row"}, "        ", reduce=False)}
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
        "count", "s_impulse[i]", "s_type[i]", op.axpy((("row", "seed"),), "        ", row="i"), op.lanes.sync(), pad=pad
    )


# ---------------------------------------------------------------------------------------------------------------------
# Contact units: a contact normal and its friction pair on their Gram block, by the row law
# ---------------------------------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class _UnitDiagonal(Diagonal):
    """A triplet unit's diagonals: the normal steps by its record's ``1/W_nn``. Every sparse row keeps its diagonal
    ``A_ii`` (``s_diag``), so the unit's rows vote in the velocity form, as the sparse rows' storage gives it (the
    storage question, option 1)."""

    def step(self, residual: str, row: str) -> str:
        return f"(r == 0 ? -{residual} * unit_inverse : -{residual} / {self.diagonal(row)})"


def _record(width: int | str, row: str) -> str:
    """The address of the contact record of the triplet that holds row ``row``, records ``width`` floats apart."""
    return f"contact_records.data + (static_cast<size_t>(line) * contact_records.shape[1] + {row} / 3) * {width}"


def contact_units(sweep: SparseSweep, rows: StagedRows, op: SparseFactor, laws: RowLaws, pad: str) -> tuple[str, str]:
    """The contact unit on the triplet's Gram block (:func:`.sweep.contact_unit`): its setup (linked: the rings'
    records), and its visit of row ``i``.

    A contact normal followed by its two tangents in the staged metadata loads in one group: the three rows'
    entries (every stored entry, then masked by its coordinate), the triplet's block record (``contact_records``:
    the Gram couplings ``W_nt1, W_nt2, W_t1t2``, ``1/W_nn`` and the pair's eigenvalues and eigenbasis) and its
    friction coefficient; one reduction gives the three rows' velocity terms. The rows then run the
    row law (:class:`.sweep.UnitZ`): the normal steps with the record's ``1/W_nn``, the pair solves at the new load from
    the record's metric, its residuals moved by the normal change through ``W_nt`` (in exact arithmetic the sequential
    sweep); each coordinate takes the sum of the normal's, the first tangent's and the second tangent's changes, added
    once. With ``linked`` a pass at entry stores each anchor's other ring members, at most three, behind
    its record in the order the pair update pools them (-1: a longer ring, which keeps the walk). A row the guards
    decline keeps the row step (:func:`.sweep.row_step`).

    Under contact torsion a contact with a spin row is a friction triple, its pair staged as spin rows
    (:class:`.sweep.StagedRows`): the guards decline it, its normal takes the row step and the triple phase its pair
    and spin row (:func:`triple_phase`). With ``linked`` the entry pass stores a triple's ring members too, which its
    visit pools.
    """
    width = CONTACT_RECORD_FLOATS + (CONTACT_RING_FLOATS if sweep.linked_rows else 0)
    pair_rows = f"s_type[normal + 1] != {FRICTION} || s_type[normal + 2] != {FRICTION}"
    if laws.torsion:
        # A triple's pair is staged as spin rows (StagedRows' triples).
        pair_rows = f"s_type[normal + 1] != s_type[normal + 2] || (s_type[normal + 1] != {FRICTION} && s_type[normal + 1] != {TORSION})"
    setup = ""
    if sweep.linked_rows:
        setup = f"""// Each anchor's other ring members, at most three, in the order the pair update pools them; -1 marks a
// longer ring, which keeps the walk. Visits read them with the record.
for (int normal = lane; normal + 2 < count; normal += {sweep.lanes}) {{
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
    entry_guard = "select_active && " if sweep.active_rows else ""
    selected = (
        "\n        & (((s_selected_words[(i + 1) / 32] >> ((i + 1) & 31)) & 1u) != 0u)" if sweep.active_rows else ""
    )
    normal_parent = "" if sweep.linked_rows else "\n    const int normal_parent = world_row_parent.data[offset + i];"
    point_guard = "" if sweep.linked_rows else " && normal_parent < 0"
    ring_read = (
        f"\n        const int4 ring = *reinterpret_cast<const int4*>(record + {CONTACT_RECORD_FLOATS});"
        if sweep.linked_rows
        else ""
    )

    def ring_load(parent: str, load: str) -> str:
        # A point contact's pool is its normal's load.
        return load

    ring = ""
    if sweep.linked_rows:
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
        # The unit's three rows share the normal's support: one reduction of their velocity terms, the support, the
        # velocity there and their entries kept for the apply.
        unit_rows = {"normal": "i", "tangent": "tangent", "sibling": "sibling"}
        sums = (("unit_jv0", "normal", None), ("unit_jv1", "tangent", None), ("unit_jv2", "sibling", None))
        gather = op.dot(sums, unit_rows, "        ", keep_velocity=True, reduce=False)
        # Per coordinate the unit's changes summed in its rows' order, then added.
        changes = (("normal", "unit_dn"), ("tangent", "unit_dt1"), ("sibling", "unit_dt2"))
        apply = op.axpy(changes, "            ", kept=True, block=True)
        ctx = RowContext(
            row="i",
            kind="row_type",
            # The unit's one reduction gathers the three rows' entries; its Gram couplings and the pair's metric are
            # the triplet's block record.
            operator=UnitZ(cached=True, fused_pair_apply=op.fused_pair_apply),
            diagonal=_UnitDiagonal(rows.diag),
            load=rows.load,
            store=rows.store,
            rhs=rows.rhs_of,
            mu=lambda row: "mu",
            weight=rows.weight,
            eps=rows.eps,
            parent="parent",
            ring_load=ring_load,
            lanes=rows.lanes,
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


def triple_phase(sweep: SparseSweep, rows: StagedRows, op: SparseFactor, laws: RowLaws, pad: str) -> str:
    """The friction triples after a sweep's rows (contact torsion), in row order, each visited by
    :func:`.sweep.friction_triple_visit`: one reduction and one apply of its pair and spin row together, on the pooled
    normal load of its contact's patch ring. A triple's normal is a contact whose next row is staged as a spin row (its
    pair, :class:`.sweep.StagedRows`): the lanes find them together, a lane a row.

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
    if sweep.linked_rows and sweep.contact_triplets:
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
        operator=op,
        diagonal=Diagonal(rows.diag),
        load=rows.load,
        store=rows.store,
        rhs=rows.rhs_of,
        mu=rows.mu,
        weight=rows.weight,
        parent="head",
        ring_load=rows.ring_load,
        lanes=rows.lanes,
    )
    # A triple's normal is a contact whose next row is staged as a spin row; only a world whose staging found a
    # triple (world_triples, uniform over its lanes) scans for them.
    heads = BallotHeads(
        "count",
        rows.lanes,
        lambda row: f"{row} + 1 < count && {rows.kind(row)} == {CONTACT} && {rows.kind(f'{row} + 1')} == {TORSION}",
        guard="world_triples",
    )
    visit = friction_triple_visit(
        ctx, laws, head="head", metric="\n".join(metric), pool="\n".join(pool), load="pooled", law=law, pad=""
    )
    return heads.wrap(indent(visit, heads.inner(pad)), pad)


# ---------------------------------------------------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------------------------------------------------


def _staging(sweep: SparseSweep, rows: StagedRows, op: SparseFactor) -> Staging:
    """Each world's shared working set: its rows and delta, the active rows' bitsets and the dense rows of
    ``local_factors``, one array per buffer for the block's worlds."""
    buffers = [Buffer("s_delta", "float", sweep.max_world_dofs), *rows.buffers()]
    if sweep.active_rows:
        words = (sweep.max_constraints + 31) // 32
        buffers += [Buffer("s_normal_words", "unsigned", words), Buffer("s_selected_words", "unsigned", words)]
    if op.dense:
        buffers.append(Buffer(op.local, "float", sweep.max_constraints * op.dense))
    return Staging(tuple(buffers), groups=sweep.groups, threads=sweep.threads, layout="arrays")


def _source(sweep: SparseSweep) -> tuple[str, Staging]:
    """The sweep's native body: staging and setup, then per sweep the active scan, the row loop (contact unit, then
    the row step), the friction triples (contact torsion), the last-word velocity-limit pass and the exit;
    then the publication."""
    M, D, S, L = sweep.max_constraints, sweep.max_world_dofs, sweep.max_row_dofs, sweep.lanes
    lanes = Lanes(L)
    rows = StagedRows(M, lanes, triples=sweep.torsion)
    op = factor_operator(sweep)
    laws = sweep.laws
    staging = _staging(sweep, rows, op)

    setup = []
    if op.dense:
        setup.append(op.expand("count", "    "))
    setup.append("    SYNC();")
    if sweep.active_rows:
        setup.append(active_selection(sweep, "    "))
    if sweep.warm_start:
        setup.append(warm_start_seed(op, "    "))
    # The row loop's enumeration: the rows the active scan selected, or every row.
    order = SelectedRows("count") if sweep.active_rows else SlotOrder("count")
    body_pad = order.inner("        ")
    unit = ""
    if sweep.contact_triplets:
        unit_setup, unit = contact_units(sweep, rows, op, laws, body_pad)
        if unit_setup:
            setup.append(unit_setup)

    scan = []
    if sweep.active_rows:
        scan.append(active_scan(sweep, op, "        "))
    if sweep.contact_triplets:
        scan.append("        int consumed_tangent = -1;")
    # Velocity-limit rows wait for the last-word pass; under contact torsion the triples are the triple phase's.
    admission = f"if (row_type == {VELOCITY_LIMIT}) continue;" if sweep.velocity_limits else ""
    update = row_step(
        rows, op, laws, end="count", admission=admission, held=True, triple_phase=sweep.torsion, pad=body_pad
    )
    row_loop = order.wrap(f"{unit}\n{update}", "        ")
    triples = triple_phase(sweep, rows, op, laws, "        ") if sweep.torsion else ""
    passes = ""
    if sweep.velocity_limits:
        passes = velocity_limit_pass(rows, op, laws, end="count", pad="        ")
    exit_ = end_sweep(
        sweep.solve_policy, iteration=ITERATION, violation=active_violation(sweep, op) if sweep.active_rows else None
    )
    sweeps = level_loop((*scan, row_loop, triples), (passes,), exit=exit_)
    body = f"""    const int lane = LANE;
    const int line = dense_first.data[world];
    __builtin_assume(line >= 0);
    const int offset = line * {M};
    const int row_base = offset * {S};
    const int count = min(world_constraint_count.data[world], {M});
{staging.declarations("    ")}
{rows.rings("    ")}

    for (int d = lane; d < {D}; d += {L}) s_delta[d] = 0.0f;
{rows.loads("    ")}
{chr(10).join(setup)}

{sweeps}

    for (int d = lane; d < {D}; d += {L}) factor_velocity_delta.data[world * {D} + d] = s_delta[d];
{rows.stores("    ")}
"""
    return sweep.native(body, lanes=L), staging

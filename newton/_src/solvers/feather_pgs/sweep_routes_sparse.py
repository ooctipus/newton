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
  a bitset (:func:`.sweep.active_scan`); patch rings (``linked_rows``) select their friction pairs through the ring; under
  contact ``torsion`` the friction triples (a contact's pair and its spin row) in a phase of their own after each
  sweep's rows (:func:`.sweep.triple_phase`), so the row loop carries none of their code;
- the **contact unit** (``contact_triplets``): a contact normal and its friction pair in one visit on their Gram
  block (:func:`contact_units`): one load group with the contact's record, one reduction of the three residuals,
  the normal step, the pair's residuals moved by the normal change through the record's couplings, the pair law from
  the record's metric, and one apply;
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
from .row_laws import ROW_CONTACT, ROW_JOINT_VELOCITY_LIMIT, ROW_TORSION
from .sparse_mass_matrix import _SparseRowSupport
from .sweep import (
    FORMED_RECORD,
    ITERATION,
    Diagonal,
    RowContext,
    RowLaws,
    SelectedRows,
    SlotOrder,
    StagedRows,
    SweepConfig,
    UnitZ,
    active_scan,
    active_selection,
    active_violation,
    active_words,
    cached_pair_law,
    end_sweep,
    formed_friction_laws,
    friction_metric,
    impulse_seed,
    level_loop,
    row_step,
    triple_phase,
    unit_visit,
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
    - the enumeration: ``active_rows`` visits the rows selected at sweep entry (:func:`.sweep.active_scan`);
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

    @property
    def records(self) -> bool:
        """Contact units read their contacts' records, and where spin rows may appear every triple its metric: the
        support writer forms them (:func:`.walk_rows.contact_record`)."""
        return self.contact_triplets or self.torsion

    def operands(self) -> tuple[RowArgument, ...]:
        return (
            *((RowArgument("friction_metric", wp.array3d[float], after="omega"),) if self.records else ()),
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


def warm_start_seed(op: SparseFactor, pad: str) -> str:
    """Give the delta the response of the impulses the row buffer carries, before the first sweep
    (:func:`.sweep.impulse_seed`)."""
    return impulse_seed(
        "count", "s_impulse[i]", "s_type[i]", op.axpy((("row", "seed"),), "        ", row="i"), op.lanes.sync(), pad=pad
    )


# ---------------------------------------------------------------------------------------------------------------------
# Contact units: a contact normal and its friction pair on their Gram block, by the row law
# ---------------------------------------------------------------------------------------------------------------------


def contact_units(sweep: SparseSweep, rows: StagedRows, op: SparseFactor, laws: RowLaws, pad: str) -> str:
    """The contact unit (:func:`.sweep.unit_visit`) of row ``i`` on its contact's record, formed by the support writer
    (:func:`.walk_rows.contact_record`: the Gram couplings ``W_nt1, W_nt2`` and the pair's metric); the normal steps by
    its staged reciprocal. A linked unit pools its patch ring by the ring's walk; otherwise only point contacts are
    units. Under contact torsion a friction triple (its pair staged as spin rows) is declined: its normal takes the
    row step and the triple phase its pair and spin row (:func:`.sweep.triple_phase`).
    """
    normal_parent = "" if sweep.linked_rows else "\n    const int normal_parent = world_row_parent.data[offset + i];"
    point_guard = "" if sweep.linked_rows else " && normal_parent < 0"
    ctx = RowContext(
        row="i",
        kind="row_type",
        operator=UnitZ(cached=True, fused_pair_apply=op.fused_pair_apply),
        diagonal=Diagonal(rows.diag, rows.inverse),
        load=rows.load,
        store=rows.store,
        rhs=rows.rhs_of,
        mu=lambda row: "mu",
        weight=rows.weight,
        eps=rows.eps,
        parent="parent",
        # A point contact's pool is its normal's load.
        ring_load=rows.ring_load if sweep.linked_rows else (lambda parent, load: load),
        lanes=rows.lanes,
        pair_law="UNIT_PAIR(0, ",
    )
    loads = f"""        const float4* const record = reinterpret_cast<const float4*>({friction_metric("i", rows=3)});
        const float4 coupling = record[0], basis = record[1];
"""
    after = "\n        const float gram_n1 = coupling.x, gram_n2 = coupling.y;\n" + cached_pair_law(
        "UNIT_PAIR", "basis.x", "basis.y", ("basis.z", "basis.w"), "        "
    )
    guard = (
        f"""
    // The guards load together; the rows, the record and the coordinates then load together.
    const int row_class = support.row_class.data[offset + i];
    const int tangent_class = support.row_class.data[offset + tangent];
    const int sibling_class = support.row_class.data[offset + sibling];
    const int tangent_parent = world_row_parent.data[offset + tangent];
    const int sibling_parent = world_row_parent.data[offset + sibling];{normal_parent}""",
        f"""tangent_parent == i && sibling_parent == i && tangent_class == sibling_class
        && row_class == tangent_class{point_guard}""",
    )
    return unit_visit(
        rows,
        op,
        laws,
        ctx,
        count="count",
        active=sweep.active_rows,
        mu="world_row_mu.data[offset + tangent]",
        guard=guard,
        record=(loads, after),
        pad=pad,
    )


# ---------------------------------------------------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------------------------------------------------


def _staging(sweep: SparseSweep, rows: StagedRows, op: SparseFactor) -> Staging:
    """Each world's shared working set: its rows and delta, the active rows' bitsets and the dense rows of
    ``local_factors``, one array per buffer for the block's worlds."""
    buffers = [Buffer("s_delta", "float", sweep.max_world_dofs), *rows.buffers()]
    if sweep.active_rows:
        buffers += active_words(sweep.max_constraints)
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
    if sweep.torsion:
        # A triple's law on its contact's record, which the support writer formed (walk_rows.contact_record).
        setup.append(formed_friction_laws("FRICTION", "    "))
    if sweep.active_rows:
        setup.append(active_selection(rows, count="count", linked=sweep.linked_rows, pad="    "))
    if sweep.warm_start:
        setup.append(warm_start_seed(op, "    "))
    # The row loop's enumeration: the rows the active scan selected, or every row.
    order = SelectedRows("count") if sweep.active_rows else SlotOrder("count")
    body_pad = order.inner("        ")
    unit = contact_units(sweep, rows, op, laws, body_pad) if sweep.contact_triplets else ""

    scan = []
    if sweep.active_rows:
        scan.append(
            active_scan(
                rows,
                op,
                count="count",
                seeded=sweep.warm_start,
                linked=sweep.linked_rows,
                torsion=sweep.torsion,
                parallel=False,
                pad="        ",
            )
        )
    if sweep.contact_triplets:
        scan.append("        int consumed_tangent = -1;")
    # Velocity-limit rows wait for the last-word pass; under contact torsion the triples are the triple phase's.
    admission = f"if (row_type == {ROW_JOINT_VELOCITY_LIMIT}) continue;" if sweep.velocity_limits else ""
    update = row_step(
        rows, op, laws, end="count", admission=admission, held=True, triple_phase=sweep.torsion, pad=body_pad
    )
    row_loop = order.wrap(f"{unit}\n{update}", "        ")
    triples = ""
    if sweep.torsion:
        # A triple reads its contact's record (the general sweep's triple phase); its normal is a contact whose next
        # row is staged as a spin row, and only a world whose staging found a triple scans for them.
        loaded = ", ".join(f"{name} = friction_record[{k}]" for k, name in enumerate(FORMED_RECORD))
        triples = triple_phase(
            rows,
            op,
            laws,
            count="count",
            record=f"const float4* const friction_record = reinterpret_cast<const float4*>({friction_metric('head', rows=3)});\n"
            f"const float4 {loaded};",
            law=f"FRICTION({', '.join(FORMED_RECORD)}, ",
            heads=lambda row: (
                f"{row} + 1 < count && {rows.kind(row)} == {ROW_CONTACT} && {rows.kind(f'{row} + 1')} == {ROW_TORSION}"
            ),
            guard="world_triples",
            pad="        ",
        )
    passes = ""
    if sweep.velocity_limits:
        passes = velocity_limit_pass(rows, op, laws, end="count", pad="        ")
    exit_ = end_sweep(
        sweep.solve_policy,
        iteration=ITERATION,
        violation=active_violation(rows, op, count="count", parallel=False) if sweep.active_rows else None,
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

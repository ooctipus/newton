# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The native sparse PGS sweep as a configuration of the sweep emitter (:mod:`.sweep`).

The sweep solves a world's rows in factor coordinates: row ``i`` is ``Z_i = L^-1 P J_i^T``, packed as the
coordinates and values of its support (:class:`.response_operator.SparseFactor`), and the velocity it acts on is
the factor-coordinate delta from ``v_hat``, staged per world and published once (``factor_velocity_delta``); the
caller decodes it. A configuration chooses:

- the **mapping**: 32, 16 or 8 lanes per world, ``64 / lanes`` worlds per block of 64 threads, with launch bounds
  from the data (:func:`launch_bounds`);
- the **operator's layout**: packed rows read through explicit row indices (``row_dof``) or the shared support
  table, or (``local_factors``) packed rows expanded once per call into dense rows in shared memory;
- the **schedule**: every row in slot order, or (``active_rows``) the rows selected at sweep entry, popped from
  a bitset (:func:`active_scan`); patch rings (``linked_rows``) select their friction pairs through the ring;
- the **contact triplet transaction** (``contact_triplets``): a contact normal and its friction pair in one visit,
  ``"staged"`` (the normal's reduction, then the pair's, with a cache of pair metrics) or ``"single"`` (one load
  group and one reduction, from the triplet's block record);
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

from . import kernel_template
from .device_preludes import RING, SHARED_STORAGE, cuda_native, lane_group
from .kernel_template import KernelTemplate
from .law_source import law_preludes
from .projection_laws import PROJECTION_LAW
from .response_operator import Lanes, SparseFactor
from .row_inputs import RowArgument
from .solve_policy import DEFAULT_POLICY, Change, SolvePolicy, bound
from .sparse_contact import CONTACT_RECORD_FLOATS, CONTACT_RING_FLOATS
from .sparse_mass_matrix import _SparseRowSupport
from .staging import Buffer, Staging, arch_budget, resident_blocks
from .sweep import (
    CONTACT,
    FRICTION,
    JOINT_LIMIT,
    VELOCITY_LIMIT,
    RowContext,
    RowLaws,
    change_vote,
    end_sweep,
    indent,
    row_visit,
    velocity_limit_pass,
)
from .velocity_limit_law import check_velocity_limit_law

_SPARSE_CONTACT_METRIC_CACHE_SIZE = 12
_SPARSE_LOCAL_WORKSPACE_BYTES = 8 * 1024
CONTACT_TRANSACTIONS = ("auto", "single", "staged")
"""Triplet transactions of the sparse sweep; ``auto`` is ``single`` for linked rows, else ``staged``."""
REGISTER_BLOCKS = 16
"""Resident blocks per SM within the sparse sweeps' register budget: 64 registers a thread, the register file's 64K
over 16 blocks of 64 threads. Every sparse configuration compiles within it without local memory."""


SPARSE_ABI = kernel_template.params(
    ("world", int),
    ("world_constraint_count", wp.array[int]),
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
    ("mf_impulses", wp.array2d[float]),
    ("mf_Z_a", wp.array3d[float]),
    ("mf_Z_b", wp.array3d[float]),
    ("mf_row_mu", wp.array2d[float]),
    ("iterations", int),
    ("omega", float),
    ("friction_start_iteration", int),
    ("iteration_offset", int),
    ("factor_velocity_delta", wp.array2d[float]),
    ("support", _SparseRowSupport),
)
"""The sparse sweep's native ABI (one world per call). ``world_dof_indices``, ``v_hat``, the ``mf_*`` operands and
the factory's free-body capacities are read by no configuration: the sweep holds no free-body rows."""

THREADS = 64
"""Threads per block: ``64 / lanes`` worlds."""


def launch_bounds(staging: Staging, device_arch=None) -> tuple[int, int]:
    """The sparse sweep's launch bounds: 64 threads, and as many resident blocks per SM as its working set's shared
    memory admits on ``device_arch`` (:func:`.staging.resident_blocks`), at most :data:`REGISTER_BLOCKS`.

    ptxas fits the sweep's registers to that occupancy, so a configuration whose shared memory allows 16 blocks keeps
    them however many registers its laws hold live, and one that shared memory limits keeps its registers.
    """
    return THREADS, min(REGISTER_BLOCKS, resident_blocks(staging.bytes, device_arch))


# ---------------------------------------------------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class SparseSweep:
    """A sparse sweep's configuration, as the factory's options resolve.

    Contact triplets need the support table, three rows, and the active-row scan or linked rows; linked rows
    take part only in an active-row scan or a triplet transaction; ``"auto"`` resolves to the single-stage
    transaction for linked rows and to the staged one otherwise; only a sweep with velocity-limit rows has a
    velocity-limit law to choose.
    """

    rows: int
    dofs: int
    support: int
    lanes: int
    active: bool
    table: bool
    triplets: bool
    linked: bool
    transaction: str
    velocity_limits: bool
    accumulated: bool
    local: bool
    warm_start: bool
    policy: SolvePolicy

    @property
    def groups(self) -> int:
        """Worlds per block."""
        return THREADS // self.lanes

    @property
    def single_stage(self) -> bool:
        """Contact triplets take the single-stage transaction (and its records)."""
        return self.triplets and self.transaction == "single"

    @property
    def staged(self) -> bool:
        """Contact triplets take the staged transaction (and its metric cache)."""
        return self.triplets and self.transaction == "staged"

    @property
    def name(self) -> str:
        """The kernel's name."""
        suffix = "_active" if self.active else ""
        suffix += "_support_table" if self.table else ""
        suffix += "_contact_triplets" if self.triplets else ""
        suffix += "_single" if self.single_stage else ""
        suffix += "_linked" if self.linked else ""
        suffix += "_velocity_limits" if self.velocity_limits else ""
        suffix += "_vacc" if self.accumulated else ""
        suffix += f"_lanes{self.lanes}" if self.lanes != 32 else ""
        suffix += "_local_factors" if self.local else ""
        suffix += "_warm_start" if self.warm_start else ""
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
    velocity_limit_law: str = "stateless",
    solve_policy: SolvePolicy = DEFAULT_POLICY,
    contact_transaction: str = "auto",
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
    if contact_transaction not in CONTACT_TRANSACTIONS:
        raise ValueError(f"contact_transaction must be one of {CONTACT_TRANSACTIONS}")
    accumulated = check_velocity_limit_law(velocity_limit_law) == "accumulated" and bool(velocity_limits)
    active = bool(active_rows)
    triplets = bool(contact_triplets) and (active or bool(linked_rows)) and bool(use_support_table) and M >= 3
    linked = bool(linked_rows) and (active or triplets)
    if contact_transaction == "auto":
        contact_transaction = "single" if linked else "staged"
    return SparseSweep(
        rows=M,
        dofs=D,
        support=S,
        lanes=L,
        active=active,
        table=bool(use_support_table),
        triplets=triplets,
        linked=linked,
        transaction=contact_transaction,
        velocity_limits=bool(velocity_limits),
        accumulated=accumulated,
        local=bool(local_factors),
        warm_start=bool(warm_start),
        policy=solve_policy,
    )


# ---------------------------------------------------------------------------------------------------------------------
# Row store and operator
# ---------------------------------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class SparseRows:
    """A world's rows in the sparse sweep.

    Impulses, right-hand sides (``rhs_bias + row_incident``), diagonals and types are staged per world
    (``s_impulse``, ``s_rhs``, ``s_diag``, ``s_type``); every lane reads them and the first lane stores the
    impulses. Parents and friction coefficients stream from global memory; a patch ring walks at most ``count``
    rows.
    """

    capacity: int
    lanes: Lanes

    def load(self, row: str) -> str:
        return f"s_impulse[{row}]"

    def store(self, row: str, value: str) -> str:
        return self.lanes.first(f"s_impulse[{row}] = {value};")

    def rhs_of(self, row: str) -> str:
        return f"s_rhs[{row}]"

    def diag(self, row: str) -> str:
        return f"s_diag[{row}]"

    def kind(self, row: str, *, scan: bool = False) -> str:
        return f"s_type[{row}]"

    def parent(self, row: str) -> str:
        return f"world_row_parent.data[offset + {row}]"

    def mu(self, row: str) -> str:
        return f"world_row_mu.data[offset + {row}]"

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
            Buffer("s_type", "unsigned char", rows),
        )

    def loads(self, pad: str) -> str:
        lane, width = self.lanes.lane, self.lanes.width
        return (
            f"{pad}for (int i = {lane}; i < count; i += {width}) {{\n"
            f"{pad}    s_impulse[i] = world_impulses.data[offset + i];\n"
            f"{pad}    s_rhs[i] = rhs_bias.data[offset + i] + row_incident.data[offset + i];\n"
            f"{pad}    s_diag[i] = world_diag.data[offset + i];\n"
            f"{pad}    s_type[i] = static_cast<unsigned char>(world_row_type.data[offset + i]);\n"
            f"{pad}}}"
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
    negative (or not finite), every other non-friction row always; a delayed friction row is zeroed. A first
    tangent is selected while its normal is selected or its pair holds impulse, a patch pair while any normal of
    its ring is. A row a sweep's earlier updates violate waits for the next sweep.
    """
    serial = dataclasses.replace(op, lanes=Lanes(1, "0"))
    # The delta is zero at the first sweep unless warm starting seeded it.
    dot = serial.accumulate("dot", "row", "")
    if not sweep.warm_start:
        dot = "if (iter != 0) {\n" + indent(dot, "    ") + "\n}"
    if sweep.linked:
        tangent = (
            "const auto normal_selected = [&](int r) { return ((s_normal_words[r / 32] >> (r & 31)) & 1u) != 0u; };\n"
            "selected = row == parent + 1 && (s_impulse[row] != 0.0f || s_impulse[parent + 2] != 0.0f\n"
            "    || (parent >= 0 && (normal_selected(parent)\n"
            "        || RING_ANY(parent, count, ring_parent, normal_selected))));"
        )
    else:
        tangent = (
            "selected = row == parent + 1 && (\n"
            "    ((s_normal_words[parent / 32] >> (parent & 31)) & 1u) != 0u\n"
            "    || s_impulse[row] != 0.0f || s_impulse[parent + 2] != 0.0f);"
        )
    lane_zero = op.lanes.first
    text = f"""for (int first = 0; first < count; first += 32) {{
    const int row = first + lane;
    bool selected = false;
    if (row < count) {{
        const int kind = s_type[row];
        if (kind != {FRICTION}) {{
            selected = !select_active || (kind != {CONTACT} && kind != {JOINT_LIMIT}) || s_impulse[row] != 0.0f;
            if (!selected && s_diag[row] > 0.0f) {{
                float dot = 0.0f;
{indent(dot, "                ")}
                const float residual = dot + s_rhs[row];
                selected = residual < 0.0f || !isfinite(residual);
            }}
        }} else if (select_active && ProjectionLaw::friction_gate(kind, global_iter, friction_start_iteration)) {{
            // The active list visits only first tangents; clear both delayed rows here.
            s_impulse[row] = 0.0f;
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
        if (select_active && s_type[row] == {FRICTION} && global_iter >= friction_start_iteration) {{
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
    """Give the delta the response of the impulses the row buffer carries, before the first sweep, as the
    ordinary route installs them in the velocity; velocity-limit rows are seeded none. A row's coordinates are
    distinct, so its lanes' updates never collide; rows are serialized by the barrier."""
    lines = [
        "for (int i = 0; i < count; ++i) {",
        "    const float seed = s_impulse[i];",
        f"    if (seed != 0.0f && s_type[i] != {VELOCITY_LIMIT}) {{",
        op.apply_row("i", "seed", "        "),
        "    }",
        f"    {op.lanes.sync()}",
        "}",
    ]
    return indent("\n".join(lines), pad)


def ordinary_rows(
    rows: SparseRows, op: SparseFactor, laws: RowLaws, *, skip_velocity_limits: bool, metric_cache: bool, pad: str
) -> str:
    """Row ``i``'s visit and update under the shared row law (:func:`.sweep.row_visit`).

    The friction gate, then the first tangent of a pair on its ring's disk; the residual reduces the pair's
    second-tangent term and coupling with it (:meth:`.SparseFactor.residual`). Velocity-limit rows wait for the
    last-word pass (``skip_velocity_limits``). Under the staged triplet transaction a pair whose metric is cached
    (``metric_cache``) skips its coupling's sum and reduction and takes the cached metric's pair law. The first
    lane stores the impulses after every lane has read them.
    """
    lanes = rows.lanes
    op = dataclasses.replace(op, diagonal=lambda row: "denom" if row == "i" else rows.diag(row))
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
        # No contact regularization: the laws read no weight.
        weight=lambda row: "1.0f",
        parent="parent",
        ring_load=rows.ring_load,
        lanes=lanes,
        fenced=False,
        pair_law="friction_pair_cached(metric_slot, " if metric_cache else RowContext.pair_law,
    )
    head = [
        f"int row_type = {rows.kind('i')};",
        f"if (row_type == {VELOCITY_LIMIT}) continue;" if skip_velocity_limits else "",
    ]
    metric = (
        f"const int metric_slot = row_type == {FRICTION} ? contact_metric_slot(parent) : -1;\n"
        "const bool metric_ready = metric_slot >= 0 && s_metric_map[metric_slot] >= 0;"
        if metric_cache
        else ""
    )
    residual = op.residual(
        "jv",
        pad,
        sibling="parent + 2",
        paired=f"row_type == {FRICTION}",
        coupling="!metric_ready" if metric_cache else None,
    )
    visit = row_visit(
        ctx,
        laws,
        parent=rows.parent("i"),
        check="denom",
        declare=f"float denom = {rows.diag('i')};",
        hold=f"const float row_impulse = {rows.load('i')};\nfloat pair_impulse = 0.0f;",
        keep_next="pair_impulse",
        entries=metric,
        residual=residual,
        pad=pad,
    )
    return indent("\n".join(line for line in head if line), pad) + "\n" + visit


# ---------------------------------------------------------------------------------------------------------------------
# Contact triplet transactions: a contact normal and its friction pair in one visit
# ---------------------------------------------------------------------------------------------------------------------


def _triplet_vote(laws: RowLaws, pad: str, *, normal_inverse: bool) -> str:
    """A triplet's vote under the exit law: the normal's change and the pair's (:func:`_tangent_state`), each in its
    storage's form: the tangents' and (unless ``normal_inverse``, the record's ``1 / W_nn``) the normal's from their
    staged diagonals."""
    changes = (
        Change("normal_change", "normal_bound", str(CONTACT), "s_diag[i]", scaled=not normal_inverse),
        Change("tangent_change", "tangent_bound", str(FRICTION), "s_diag[tangent]", scaled=True),
        Change("sibling_change", "sibling_bound", str(FRICTION), "s_diag[sibling]", scaled=True),
    )
    return change_vote(laws, changes, pad)


def _triplet_pair(*, reduce: str, pair_law: str, apply: str) -> str:
    """The pair's update at the new normal load: ``reduce`` its residuals, project the pair on its disk by
    ``pair_law``, store it from the first lane (every lane read the pair's impulses before the normal's store) and
    ``apply`` its changes to the coordinates a transaction holds. A zero disk without carried impulse is skipped."""
    lines = [
        "if (!(radius == 0.0f && old_tangent == 0.0f && old_sibling == 0.0f)) {",
        indent(reduce, "    ") if reduce else "",
        "    wp::vec2 pair(0.0f);",
        "    if (radius > 0.0f) {",
        indent(pair_law, "        "),
        "    }",
        "    pair = ProjectionLaw::disk(pair, radius);",
        "    const float new_tangent = pair[0], new_sibling = pair[1];",
        "    tangent_change = new_tangent - old_tangent;",
        "    sibling_change = new_sibling - old_sibling;",
        "    if (lane == 0) { s_impulse[tangent] = new_tangent; s_impulse[sibling] = new_sibling; }",
        indent(apply, "    ") if apply else "",
        "}",
    ]
    return "\n".join(line for line in lines if line)


def _tangent_state(bounds: tuple[str, str] = ("0.0f", "0.0f")) -> str:
    """The pair's changes (zero for a skipped pair) and their bounds under the exit law (``bounds``, or set where the
    pair reduces its residuals; a skipped pair changes nothing)."""
    tangent_bound, sibling_bound = bounds
    return (
        "float tangent_change = 0.0f, sibling_change = 0.0f, "
        f"tangent_bound = {tangent_bound}, sibling_bound = {sibling_bound};"
    )


def _pair_bounds(tangent_dot: str, sibling_dot: str, tangent_rhs: str, sibling_rhs: str) -> tuple[str, str]:
    """The bounds of a triplet's tangent pair from the reduced sums of its residuals, at its old impulses."""
    return (
        bound(tangent_rhs, f"fabsf({tangent_dot})", "old_tangent", diagonal="s_diag[tangent]"),
        bound(sibling_rhs, f"fabsf({sibling_dot})", "old_sibling", diagonal="s_diag[sibling]"),
    )


def _record(width: int, row: str) -> str:
    """The address of the contact record of the triplet that holds row ``row``."""
    return f"contact_records.data + (static_cast<size_t>(world) * contact_records.shape[1] + {row} / 3) * {width}"


def staged_transaction(
    sweep: SparseSweep, rows: SparseRows, op: SparseFactor, laws: RowLaws, pad: str
) -> tuple[str, str]:
    """The staged triplet transaction: the setup of its metric cache, and its visit of row ``i``.

    A contact normal with its two tangents (same support class, parents the normal; unlinked: a point contact)
    reduces the normal's residual over its support, takes the normal step, moves the coordinates it holds in
    registers by the normal's change, then reduces the pair's residuals from those coordinates (the pair's
    coupling only until its metric is cached) and solves the pair at the new load (``linked``: its ring pooled from
    the new normal on, in the order the pair update pools it); the coordinates are written back once. The cache
    keeps at most ``min(12, rows / 3)`` pair metrics per world and recomputes the others. The first lane stores the
    impulses after every lane read them; a row the guards decline keeps the ordinary update.
    """
    pairs = sweep.rows // 3
    cache_size = min(_SPARSE_CONTACT_METRIC_CACHE_SIZE, pairs)
    count = op.entries
    entry_guard = "!select_active || " if sweep.active else ""
    point_guard = "" if sweep.linked else "\n        || world_row_parent.data[offset + normal] >= 0"
    selected = "\n    && ((s_selected_words[(i + 1) / 32] >> ((i + 1) & 31)) & 1u) != 0u" if sweep.active else ""
    setup = f"""for (int id = lane; id < {pairs}; id += 32) s_metric_map[id] = -1;
if (lane == 0) s_metric_count[0] = 0;
SYNC();
// A full triplet's slot: normal / 3 is injective over full, disjoint three-row records, even when another row
// kind precedes a contact or a triplet crosses a mask word.
const auto contact_metric_slot = [&](int normal) {{
    if ({entry_guard}normal < 0 || normal + 2 >= count
        || s_type[normal] != {CONTACT}
        || s_type[normal + 1] != {FRICTION} || s_type[normal + 2] != {FRICTION}{point_guard}
        || world_row_parent.data[offset + normal + 1] != normal
        || world_row_parent.data[offset + normal + 2] != normal
        || support.row_class.data[offset + normal + 1] != support.row_class.data[offset + normal + 2]) return -1;
    return normal / 3;
}};
// The pair law from the pair's cached metric, or from its block, which the first lane caches while there is room
// (one group owns the world's ordered visits; a full cache recomputes).
const auto friction_pair_cached = [&](int id, float a, float c, float d,
    wp::vec2 residual, wp::vec2 old, float radius, float relaxation) {{
    float scale, largest, smallest;
    wp::vec2 axis;
    const int cached = id >= 0 ? s_metric_map[id] : -1;
    if (cached >= 0) {{
        scale = s_metrics[cached * 5];
        largest = s_metrics[cached * 5 + 1];
        smallest = s_metrics[cached * 5 + 2];
        axis = wp::vec2(s_metrics[cached * 5 + 3], s_metrics[cached * 5 + 4]);
    }} else {{
        const auto metric = ProjectionLaw::friction_pair_metric(a, c, d);
        scale = metric.v0;
        largest = metric.v1;
        smallest = metric.v2;
        axis = metric.v3;
        // Every lane has read the map before the first lane writes it.
        SYNC();
        if (id >= 0 && lane == 0 && s_metric_count[0] < {cache_size}) {{
            const int entry = s_metric_count[0];
            s_metrics[entry * 5] = scale;
            s_metrics[entry * 5 + 1] = largest;
            s_metrics[entry * 5 + 2] = smallest;
            s_metrics[entry * 5 + 3] = axis[0];
            s_metrics[entry * 5 + 4] = axis[1];
            s_metric_map[id] = static_cast<signed char>(entry);
            s_metric_count[0] = entry + 1;
        }}
        SYNC();
    }}
    return ProjectionLaw::friction_pair_from_metric(scale, largest, smallest, axis, residual, old, radius, relaxation);
}};"""
    normal_gather = op.entries_loop(
        f"""const int index = {op.guarded_coordinate("i")};
indices[q] = index;
normal_responses[q] = index >= 0 ? {op.element("i")} : 0.0f;
velocities[q] = index >= 0 ? s_delta[index] : 0.0f;
if (index >= 0) normal_dot += normal_responses[q] * velocities[q];""",
        "    ",
    )
    normal_move = op.entries_loop(
        "if (indices[q] >= 0 && normal_change != 0.0f)\n    velocities[q] += normal_responses[q] * normal_change;",
        "    ",
        entry=False,
    )
    pair_gather = op.entries_loop(
        f"""const int index = indices[q];
const float value = index >= 0 ? {op.element("tangent")} : 0.0f;
const float sibling_value = index >= 0 ? {op.element("sibling")} : 0.0f;
tangent_responses[q] = value;
sibling_responses[q] = sibling_value;
if (index >= 0) {{
    // The coordinates after the normal's change, as the ordinary pair update reads them.
    dot += value * velocities[q];
    sibling_dot += sibling_value * velocities[q];
    if (!ready) cross += value * sibling_value;
}}""",
        "",
    )
    pair_move = op.entries_loop(
        """if (indices[q] >= 0) {
    if (sibling_change != 0.0f) velocities[q] += sibling_responses[q] * sibling_change;
    if (tangent_change != 0.0f) velocities[q] += tangent_responses[q] * tangent_change;
}""",
        "",
        entry=False,
    )
    write_back = op.entries_loop("if (indices[q] >= 0) s_delta[indices[q]] = velocities[q];", "        ", entry=False)
    load, ring = "new_normal", ""
    if sweep.linked:
        load, ring = "load", f"\n    const float load = {rows.ring_load('i', 'new_normal')};"
    staged_bounds = _pair_bounds("dot", "sibling_dot", "s_rhs[tangent]", "s_rhs[sibling]")
    reduce = f"""const bool ready = s_metric_map[triplet_slot] >= 0;
float dot = 0.0f, sibling_dot = 0.0f, cross = 0.0f;
{pair_gather}
SUM(dot, sibling_dot);
if (!ready) SUM(cross);
tangent_bound = {staged_bounds[0]};
sibling_bound = {staged_bounds[1]};
const float residual = dot + s_rhs[tangent];"""
    pair_law = """const float sibling_residual = sibling_dot + s_rhs[sibling];
pair = friction_pair_cached(triplet_slot, s_diag[tangent], cross, s_diag[sibling],
    wp::vec2(residual, sibling_residual), wp::vec2(old_tangent, old_sibling), radius, omega);"""
    pair = _triplet_pair(reduce=reduce, pair_law=pair_law, apply=pair_move)
    update = f"""if (i == consumed_tangent) continue;
const int triplet_slot = contact_metric_slot(i);
const bool fuse = triplet_slot >= 0 && global_iter >= friction_start_iteration
    && isfinite(s_diag[i]) && s_diag[i] > 0.0f
    && support.row_class.data[offset + i] == support.row_class.data[offset + i + 1]{selected};
if (fuse) {{
    const int tangent = i + 1, sibling = i + 2;
    int indices[{count}];
    float velocities[{count}], normal_responses[{count}];
    float tangent_responses[{count}], sibling_responses[{count}];
    float normal_dot = 0.0f;
{normal_gather}
    SUM(normal_dot);
    const float normal_residual = normal_dot + s_rhs[i];
    const float old_normal = s_impulse[i];
    const float normal_bound = {bound("s_rhs[i]", "fabsf(normal_dot)", "old_normal", diagonal="s_diag[i]")};
    const float new_normal = ProjectionLaw::unilateral(
        ProjectionLaw::relaxed_step(old_normal, -normal_residual / s_diag[i], omega));
    const float normal_change = new_normal - old_normal;{ring}
    const float radius = fmaxf(world_row_mu.data[offset + tangent] * {load}, 0.0f);
    const float old_tangent = s_impulse[tangent], old_sibling = s_impulse[sibling];
    {_tangent_state()}
    // Every lane has read the triplet's impulses before the first lane writes them.
    SYNC();
    if (lane == 0) s_impulse[i] = new_normal;
{normal_move}
{indent(pair, "    ")}
    if (normal_change != 0.0f || tangent_change != 0.0f || sibling_change != 0.0f) {{
{_triplet_vote(laws, "        ", normal_inverse=False)}
{write_back}
    }}
    consumed_tangent = tangent;
    SYNC();
    continue;
}}"""
    return indent(setup, "    "), indent(update, pad)


def single_stage_transaction(
    sweep: SparseSweep, rows: SparseRows, op: SparseFactor, laws: RowLaws, pad: str
) -> tuple[str, str]:
    """The single-stage triplet transaction: its setup (linked: the rings' records), and its visit of row ``i``.

    A contact normal followed by its two tangents in the staged metadata loads in one group: the three rows'
    entries (every stored entry, then masked by its coordinate), the triplet's block record (``contact_records``:
    ``W_nt1, W_nt2, W_t1t2, 1/W_nn`` and the pair's eigenbasis relative to its larger tangent diagonal) and its
    friction coefficient; one reduction gives the three residuals. The normal steps with the record's ``1/W_nn``,
    the pair solves at the new load from the record's metric, its residuals moved by the normal change through
    ``W_nt``; each coordinate takes the normal's, then the second tangent's, then the first tangent's change and is
    written back once. Its iterates differ from the staged transaction's by rounding only. With ``linked`` a pass at
    entry stores each anchor's other ring members, at most three, behind its record in the order the pair update
    pools them (-1: a longer ring, which keeps the walk). The first lane stores the impulses after every lane read
    them; a row the guards decline keeps the ordinary update.
    """
    S = sweep.support
    width = CONTACT_RECORD_FLOATS + (CONTACT_RING_FLOATS if sweep.linked else 0)
    count = op.entries
    setup = ""
    if sweep.linked:
        setup = f"""// Each anchor's other ring members, at most three, in the order the pair update pools them; -1 marks a
// longer ring, which keeps the walk. Visits read them with the record.
for (int normal = lane; normal + 2 < count; normal += 32) {{
    if (s_type[normal] != {CONTACT} || s_type[normal + 1] != {FRICTION} || s_type[normal + 2] != {FRICTION}) continue;
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
    gather = op.entries_loop(
        f"""const bool stored = entry < {S};
const float normal_entry = stored ? {op.element("i")} : 0.0f;
const float tangent_entry = stored ? {op.element("tangent")} : 0.0f;
const float sibling_entry = stored ? {op.element("sibling")} : 0.0f;
const int index = stored ? {_class_coordinate("row_class", S)} : -1;
indices[q] = index;
normal_responses[q] = index >= 0 ? normal_entry : 0.0f;
tangent_responses[q] = index >= 0 ? tangent_entry : 0.0f;
sibling_responses[q] = index >= 0 ? sibling_entry : 0.0f;
velocities[q] = index >= 0 ? s_delta[index] : 0.0f;
if (index >= 0) {{
    normal_dot += normal_responses[q] * velocities[q];
    tangent_dot += tangent_responses[q] * velocities[q];
    sibling_dot += sibling_responses[q] * velocities[q];
}}""",
        "        ",
    )
    apply = op.entries_loop(
        """if (indices[q] >= 0) {
    float value = velocities[q];
    if (normal_change != 0.0f) value += normal_responses[q] * normal_change;
    if (sibling_change != 0.0f) value += sibling_responses[q] * sibling_change;
    if (tangent_change != 0.0f) value += tangent_responses[q] * tangent_change;
    s_delta[indices[q]] = value;
}""",
        "            ",
        entry=False,
    )
    load, ring = "new_normal", ""
    if sweep.linked:
        load = "load"
        ring = f"""
        // The ring pools as the pair update walks it, from the new normal on.
        float load = new_normal;
        if (ring.w >= 0) {{
            if (ring.w > 0) load += s_impulse[ring.x];
            if (ring.w > 1) load += s_impulse[ring.y];
            if (ring.w > 2) load += s_impulse[ring.z];
        }} else {{
            load = {rows.ring_load("i", "load")};
        }}"""
    pair_law = """// W_nt moves the tangent residuals by the normal change; the eigenbasis is relative to the larger tangent
// diagonal, as the pair law scales it.
pair = ProjectionLaw::friction_pair_from_metric(tangent_scale, basis.x, basis.y, wp::vec2(basis.z, basis.w),
    wp::vec2(tangent_residual + coupling.x * normal_change, sibling_residual + coupling.y * normal_change),
    wp::vec2(old_tangent, old_sibling), radius, omega);"""
    pair = _triplet_pair(reduce="", pair_law=pair_law, apply="")
    update = f"""if (i == consumed_tangent) continue;
// The staged metadata first, read together: a contact normal with its two tangents.
if ({entry_guard}i + 2 < count && global_iter >= friction_start_iteration
    && ((s_type[i] == {CONTACT}) & (s_type[i + 1] == {FRICTION}) & (s_type[i + 2] == {FRICTION})
        & (isfinite(s_diag[i]) != 0) & (s_diag[i] > 0.0f){selected})) {{
    const int tangent = i + 1, sibling = i + 2;
    // The guards load together; the rows, the record and the coordinates then load together.
    const int row_class = support.row_class.data[offset + i];
    const int tangent_class = support.row_class.data[offset + tangent];
    const int sibling_class = support.row_class.data[offset + sibling];
    const int tangent_parent = world_row_parent.data[offset + tangent];
    const int sibling_parent = world_row_parent.data[offset + sibling];{normal_parent}
    if (tangent_parent == i && sibling_parent == i && tangent_class == sibling_class
        && row_class == tangent_class{point_guard}) {{
        const float* record = {_record(width, "i")};
        const float4 coupling = *reinterpret_cast<const float4*>(record);
        const float4 basis = *reinterpret_cast<const float4*>(record + 4);{ring_read}
        const float mu = world_row_mu.data[offset + tangent];
        int indices[{count}];
        float velocities[{count}], normal_responses[{count}];
        float tangent_responses[{count}], sibling_responses[{count}];
        float normal_dot = 0.0f, tangent_dot = 0.0f, sibling_dot = 0.0f;
{gather}
        // The visit's staged values load with the rows, off the reduction's chain.
        const float old_normal = s_impulse[i];
        const float old_tangent = s_impulse[tangent], old_sibling = s_impulse[sibling];
        const float normal_rhs = s_rhs[i], tangent_rhs = s_rhs[tangent], sibling_rhs = s_rhs[sibling];
        const float tangent_scale = fmaxf(fmaxf(s_diag[tangent], s_diag[sibling]), 1.0e-20f);
        SUM(normal_dot, tangent_dot, sibling_dot);
        const float normal_residual = normal_dot + normal_rhs;
        const float tangent_residual = tangent_dot + tangent_rhs;
        const float sibling_residual = sibling_dot + sibling_rhs;
        const float normal_bound = {bound("normal_rhs", "fabsf(normal_dot)", "old_normal", step="coupling.w")};
        const float new_normal = ProjectionLaw::unilateral(
            ProjectionLaw::relaxed_step(old_normal, -normal_residual * coupling.w, omega));
        const float normal_change = new_normal - old_normal;{ring}
        const float radius = fmaxf(mu * {load}, 0.0f);
        {_tangent_state(_pair_bounds("tangent_dot", "sibling_dot", "tangent_rhs", "sibling_rhs"))}
        // Shuffles order execution, not memory: every lane has read the triplet's impulses before the first
        // lane writes them.
        SYNC();
        if (lane == 0) s_impulse[i] = new_normal;
{indent(pair, "        ")}
        if (normal_change != 0.0f || tangent_change != 0.0f || sibling_change != 0.0f) {{
{_triplet_vote(laws, "            ", normal_inverse=True)}
{apply}
        }}
        consumed_tangent = tangent;
        SYNC();
        continue;
    }}
}}"""
    return indent(setup, "    ") if setup else "", indent(update, pad)


# ---------------------------------------------------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------------------------------------------------


def _staging(sweep: SparseSweep, rows: SparseRows, op: SparseFactor) -> Staging:
    """Each world's shared working set: its rows and delta, the active rows' bitsets, the dense rows of
    ``local_factors`` and the staged transaction's metric cache, one array per buffer for the block's worlds."""
    buffers = list(rows.buffers(sweep.dofs))
    if sweep.active:
        words = (sweep.rows + 31) // 32
        buffers += [Buffer("s_normal_words", "unsigned", words), Buffer("s_selected_words", "unsigned", words)]
    if op.dense:
        buffers.append(Buffer(op.local, "float", sweep.rows * op.dense))
    if sweep.staged:
        pairs = sweep.rows // 3
        cache_size = min(_SPARSE_CONTACT_METRIC_CACHE_SIZE, pairs)
        buffers += [
            Buffer("s_metrics", "float", cache_size * 5),
            Buffer("s_metric_map", "signed char", pairs),
            Buffer("s_metric_count", "int", 1),
        ]
    return Staging(tuple(buffers), groups=sweep.groups, threads=THREADS, layout="arrays")


def sparse_staging(max_constraints: int, max_world_dofs: int, max_row_dofs: int, **options) -> Staging:
    """The shared working set of the sweep :func:`sparse_kernel` builds from the same arguments, without its body."""
    sweep = _resolve(max_constraints, max_world_dofs, max_row_dofs, **options)
    return _staging(sweep, SparseRows(sweep.rows, Lanes(sweep.lanes)), factor_operator(sweep))


def _source(sweep: SparseSweep) -> tuple[str, Staging]:
    """The sweep's native body: staging and setup, then per sweep the active scan, the row loop (transaction, then
    the ordinary update), the last-word velocity-limit pass and the exit; then the publication."""
    M, D, S, L = sweep.rows, sweep.dofs, sweep.support, sweep.lanes
    lanes = Lanes(L)
    rows = SparseRows(M, lanes)
    op = factor_operator(sweep)
    laws = RowLaws(
        drive=False,
        velocity_limit_law="accumulated" if sweep.accumulated else "stateless",
        regularize=False,
        policy=sweep.policy,
    )
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
    transaction = ""
    if sweep.triplets:
        build = single_stage_transaction if sweep.single_stage else staged_transaction
        triplet_setup, transaction = build(sweep, rows, op, laws, body_pad)
        if triplet_setup:
            setup.append(triplet_setup)

    scan = []
    if sweep.active:
        scan.append(active_scan(sweep, op, "        "))
    if sweep.triplets:
        scan.append("        int consumed_tangent = -1;")
    update = ordinary_rows(
        rows, op, laws, skip_velocity_limits=sweep.velocity_limits, metric_cache=sweep.staged, pad=body_pad
    )
    if sweep.active:
        row_loop = f"""        for (int word = 0; word < (count + 31) / 32; ++word) {{
            unsigned pending = s_selected_words[word];
            while (pending != 0u) {{
                const int i = word * 32 + FIRST(pending);
                pending &= pending - 1u;
{transaction}
{update}
            }}
        }}"""
    else:
        row_loop = f"""        for (int i = 0; i < count; ++i) {{
{transaction}
{update}
        }}"""
    passes = ""
    if sweep.velocity_limits:
        passes = velocity_limit_pass(rows, op, laws, end="count", pad="        ")
    exit_ = end_sweep(sweep.policy, iteration="iter", violation=active_violation(sweep, op) if sweep.active else None)
    body = f"""    const int lane = LANE;
    const int offset = world * {M};
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
        const int global_iter = iteration_offset + iter;
        int iteration_changed = 0;
{chr(10).join(scan)}
{row_loop}
{passes}
{exit_}    }}

    for (int d = lane; d < {D}; d += {L}) factor_velocity_delta.data[world * {D} + d] = s_delta[d];
    for (int i = lane; i < count; i += {L}) world_impulses.data[offset + i] = s_impulse[i];
#endif
"""
    preludes = (lane_group(L), SHARED_STORAGE, RING, *law_preludes(PROJECTION_LAW), *sweep.policy.preludes)
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
    velocity_limit_law: str = "stateless",
    solve_policy: SolvePolicy = DEFAULT_POLICY,
    contact_transaction: str = "auto",
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
            velocity_limit_law=velocity_limit_law,
            solve_policy=solve_policy,
            contact_transaction=contact_transaction,
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
    velocity_limit_law: str = "stateless",
    solve_policy: SolvePolicy = DEFAULT_POLICY,
    contact_transaction: str = "auto",
    device_arch=None,
) -> wp.Kernel:
    """The sparse sweep kernel.

    Launch ``ceil(world_count / (64 / lanes_per_world))`` tiles of 64 threads. The single-stage transaction takes
    ``contact_records`` after ``iteration_offset``; a solve readout appends its operand after ``support``. The
    working set must fit the static budget of ``device_arch`` (:func:`.staging.arch_budget`).
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
        velocity_limit_law=velocity_limit_law,
        solve_policy=solve_policy,
        contact_transaction=contact_transaction,
    )
    snippet, staging = _source(sweep)
    staging.check("the sparse sweep", arch_budget(device_arch))
    extra = sweep.policy.arguments
    if sweep.single_stage:
        extra = (RowArgument("contact_records", wp.array3d[float], after="iteration_offset"), *extra)
    return KernelTemplate.from_abi(
        sweep.name,
        SPARSE_ABI,
        snippet,
        kernel_template.tiles(sweep.groups, sweep.lanes),
        extra=extra,
        composed=True,
        native_name="sweep_sparse_native",
        launch_bounds=launch_bounds(staging, device_arch),
    ).kernel()

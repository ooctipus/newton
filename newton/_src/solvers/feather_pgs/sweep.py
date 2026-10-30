# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The sweep emitter: FeatherPGS's projected Gauss-Seidel and Jacobi sweeps from one generator.

A sweep is a configuration ``Sweep(mapping, staging, blocks, update, laws, exit, publication)``:

- the **mapping** says which threads own a problem (a world, or a lane group per world) and how many lanes update
  one row together (:class:`.response_operator.Lanes`);
- the **staging** declares the block's shared working set (:class:`.staging.Staging`);
- each **row block** pairs a row store (where a row's impulse, type, parent, friction coefficient, right-hand side
  and diagonal live: :class:`WorldRows`, :class:`FreeBodyRows`) with a response operator
  (:mod:`.response_operator`) and the row kinds it visits;
- the **row visit** admits a row of a row loop (:func:`row_visit`): the delayed-friction gate (:func:`friction_gate`),
  a friction pair at its first tangent (:func:`admit_pair`), the diagonal check and the pair's disk on the patch
  ring's pooled load (:func:`friction_radius`), then the operator's residual;
- the **row law** is one function, :func:`row_update`: step, regularization, the projection of each row kind
  (unilateral, friction pair, velocity limit, bilateral), the change vote (:func:`change_vote`) and the
  apply. Every block of every sweep runs it; a friction pair solved as one unit runs its pair law and tail
  (:func:`pair_update`);
- the **contact unit** is a contact normal and its friction pair visited by the row law on the unit's Gram block
  (:func:`contact_unit`, :class:`UnitOperator`): the placement reduces the unit's terms once and applies its changes
  once, the pair's terms moved by the normal's change;
- the **Jacobi law** steps every row of a world at once instead, a lane each (:class:`JacobiRows`,
  :func:`jacobi_sweeps`): the scaled step from the extrapolated impulses, the radial projection of each friction
  pair, Nesterov's momentum with adaptive restart and the relative change vote, the operator forming each row's
  residual from the impulses;
- the **exit** is the exit law (:mod:`.solve_policy`): each change votes against the rounding floor of its update
  (:func:`change_vote`) and the end of the sweep stops a world none of whose updates was loud (:func:`end_sweep`);
- the **publication** writes the impulses and velocities back.

Only the response operator and the lane layout fix a row's floating-point order, so a configuration that keeps a
sweep's operator and layout reproduces its results.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import ClassVar

from .device_preludes import DevicePrelude
from .kernels import (
    PGS_CONSTRAINT_TYPE_CONTACT,
    PGS_CONSTRAINT_TYPE_FRICTION,
    PGS_CONSTRAINT_TYPE_JOINT_LIMIT,
    PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT,
    PGS_CONSTRAINT_TYPE_TORSION,
)
from .response_operator import Lanes, ResponseOperator
from .row_metadata import _DENSE_META_ROW_TYPE_BITS, _DENSE_META_ROW_TYPE_MASK, _mf_parent_expression
from .solve_policy import DEFAULT_POLICY, Change, SolvePolicy, bound, sweep_exit, terms_bound, vote
from .staging import Buffer

CONTACT = int(PGS_CONSTRAINT_TYPE_CONTACT)
FRICTION = int(PGS_CONSTRAINT_TYPE_FRICTION)
JOINT_LIMIT = int(PGS_CONSTRAINT_TYPE_JOINT_LIMIT)
VELOCITY_LIMIT = int(PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT)
TORSION = int(PGS_CONSTRAINT_TYPE_TORSION)


def indent(text: str, prefix: str) -> str:
    """Prefix every non-empty line of ``text``."""
    return "\n".join(prefix + line if line.strip() else line for line in text.split("\n"))


CPU_MATH = DevicePrelude(
    "cpu_math",
    "    // The CUDA math the row law calls: the device's own.",
    "    // The CUDA math the row law calls, on the CPU: Warp's own maximum (its CPU build declares no fmaxf).\n"
    "    const auto fmaxf = [](float a, float b) -> float { return wp::max(a, b); };",
)
"""The CUDA math intrinsics the row law calls, for a body that also runs on the CPU
(:func:`.device_preludes.dual_native`): ``fmaxf`` is Warp's ``wp::max`` there, the same maximum up to the sign of a
zero one; CUDA has its own."""


# ---------------------------------------------------------------------------------------------------------------------
# Row stores: where a row's state and metadata live
# ---------------------------------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class WorldRows:
    """A world's dense rows (``world_*`` arrays, ``capacity`` slots per world at ``off_dense``).

    Impulses are staged in ``s_lam_dense`` and owned by the row's first lane.
    With ``shared_metadata`` the right-hand side, diagonal and friction
    coefficient are staged too; otherwise
    they stream from their global arrays. ``packed_types`` keeps the row types
    and parents resident as one packed word per row (``s_meta_dense``: the type
    in the low bits, the parent plus one above); the end-of-sweep scans
    (:meth:`kind` with ``scan``) read the global types whenever the metadata
    streams. Staged and streamed reads hold identical values.
    """

    capacity: int
    shared_metadata: bool = True
    packed_types: bool = True
    rhs: str = "rhs_bias"
    lanes: Lanes = dataclasses.field(default_factory=Lanes)

    STAGED: ClassVar[dict[str, str]] = {
        "s_rhs_dense": "rhs",
        "s_diag_dense": "world_diag",
        "s_mu_dense": "world_row_mu",
    }

    def _source(self, name: str) -> str:
        source = self.STAGED[name]
        return self.rhs if source == "rhs" else source

    def staged(self, name: str, row: str) -> str:
        """Row ``row``'s value of the staged array ``name`` (staged or streamed)."""
        if self.shared_metadata:
            return f"{name}[{row}]"
        return f"{self._source(name)}.data[off_dense + ({row})]"

    def rhs_of(self, row: str) -> str:
        return self.staged("s_rhs_dense", row)

    def diag(self, row: str) -> str:
        return self.staged("s_diag_dense", row)

    def mu(self, row: str) -> str:
        return self.staged("s_mu_dense", row)

    def weight(self, row: str) -> str:
        return f"world_row_w.data[off_dense + {row}]"

    def kind(self, row: str, *, scan: bool = False) -> str:
        if self.shared_metadata or (self.packed_types and not scan):
            return f"s_meta_dense[{row}] & {_DENSE_META_ROW_TYPE_MASK}"
        return f"world_row_type.data[off_dense + {row}]"

    def parent(self, row: str, *, enclosed: bool = False) -> str:
        if not self.packed_types:
            return f"world_row_parent.data[off_dense + {row}]"
        decoded = f"(s_meta_dense[{row}] >> {_DENSE_META_ROW_TYPE_BITS}) - 1"
        return f"({decoded})" if enclosed else decoded

    def load(self, row: str) -> str:
        """Row ``row``'s impulse on every lane (read by the first lane, which owns it)."""
        return self.lanes.broadcast(f"s_lam_dense[{row}]")

    def peek(self, row: str) -> str:
        """Row ``row``'s impulse read directly (every lane reads the staged value)."""
        return f"s_lam_dense[{row}]"

    def store(self, row: str, value: str) -> str:
        return self.lanes.first(f"s_lam_dense[{row}] = {value};")

    def ring_load(self, parent: str) -> str:
        """The pooled normal load of the patch ring of normal row ``parent``."""
        return (
            f"RING_LOAD(s_lam_dense[{parent}], {parent}, {self.capacity},\n"
            f"    [&](int patch_row) {{ return {self.parent('patch_row', enclosed=True)}; }},\n"
            f"    [&](int patch_row) {{ return s_lam_dense[patch_row]; }})"
        )

    # -- staging ---------------------------------------------------------------------------------------------------

    def staged_names(self) -> tuple[list[str], list[str]]:
        return (["s_rhs_dense", "s_diag_dense"], ["s_mu_dense"]) if self.shared_metadata else ([], [])

    def buffers(self) -> tuple[Buffer, ...]:
        head, tail = self.staged_names()
        packed = [Buffer("s_meta_dense", "int", self.capacity)] if self.packed_types else []
        return (
            Buffer("s_lam_dense", "float", self.capacity),
            *(Buffer(name, "float", self.capacity) for name in head),
            *packed,
            *(Buffer(name, "float", self.capacity) for name in tail),
        )

    def loads(self, row: str = "i") -> str:
        head, tail = self.staged_names()
        lines = [f"s_lam_dense[{row}] = world_impulses.data[off_dense + {row}];"]
        lines += [f"{name}[{row}] = {self._source(name)}.data[off_dense + {row}];" for name in head]
        if self.packed_types:
            lines += [
                f"int row_type = world_row_type.data[off_dense + {row}];",
                f"int row_parent = world_row_parent.data[off_dense + {row}];",
                f"s_meta_dense[{row}] = (row_type & {_DENSE_META_ROW_TYPE_MASK}) | "
                f"((row_parent + 1) << {_DENSE_META_ROW_TYPE_BITS});",
            ]
        lines += [f"{name}[{row}] = {self._source(name)}.data[off_dense + {row}];" for name in tail]
        return "\n".join(lines)


@dataclasses.dataclass(frozen=True)
class FreeBodyRows:
    """A world's matrix-free (free-body) rows: one packed ``int4`` per row and the impulses.

    ``mf_meta`` holds, per row, the bodies' first coordinates (``dof_a << 16 |
    dof_b``), the stored inverse diagonal, the right-hand side and the type
    and parent (``type | parent << 16``). Impulses are staged in ``s_lam_mf``
    or, with ``stream``, read and written in place by the first lane
    (``g_lam_mf``), whose reads every lane takes by broadcast.
    """

    capacity: int
    stream: bool = False
    lanes: Lanes = dataclasses.field(default_factory=Lanes)

    def parent_of(self, packed: str) -> str:
        return _mf_parent_expression(packed, self.capacity)

    def load(self, row: str) -> str:
        if self.stream:
            return self.lanes.broadcast(f"g_lam_mf[{row}]")
        return f"s_lam_mf[{row}]"

    def store(self, row: str, value: str) -> str:
        if self.stream:
            return self.lanes.first(f"g_lam_mf[{row}] = {value};")
        return f"s_lam_mf[{row}] = {value};"

    def ring_load(self, parent: str) -> str:
        meta_parent = self.parent_of("mf_meta.data[off_meta + patch_row * 4 + 3]")
        return (
            f"RING_LOAD({self.load(parent)}, {parent}, {self.capacity},\n"
            f"    [&](int patch_row) {{ return {meta_parent}; }},\n"
            f"    [&](int patch_row) {{ return {self.load('patch_row')}; }})"
        )

    def rhs_meta(self, row: str) -> str:
        return f"__int_as_float(mf_meta.data[off_meta + {row} * 4 + 2])"

    def inverse_meta(self, row: str) -> str:
        return f"__int_as_float(mf_meta.data[off_meta + {row} * 4 + 1])"

    def mu(self, row: str) -> str:
        return f"mf_row_mu.data[off_mf + {row}]"

    def weight(self, row: str) -> str:
        return f"mf_row_w.data[off_mf + {row}]"

    def buffers(self) -> tuple[Buffer, ...]:
        return () if self.stream else (Buffer("s_lam_mf", "float", self.capacity),)

    def declaration(self) -> str:
        if self.stream:
            return "    volatile float* g_lam_mf = (volatile float*)(&mf_impulses.data[off_mf]);"
        return ""

    def load_line(self, row: str = "i") -> str:
        return "" if self.stream else f"s_lam_mf[{row}] = mf_impulses.data[off_mf + {row}];"

    def store_line(self, row: str = "i") -> str:
        return "" if self.stream else f"mf_impulses.data[off_mf + {row}] = s_lam_mf[{row}];"


# ---------------------------------------------------------------------------------------------------------------------
# The row law
# ---------------------------------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class RowLaws:
    """The physics a sweep's rows obey, beyond each row kind's projection.

    Args:
        regularize: Contact rows take their regularization weight when the run-time ``regularize`` is set.
        torsion: Torsion rows exist; the row loop skips them and the torsion pass solves them.
        policy: The solve policy: the readout beside the exit law.
    """

    regularize: bool = True
    torsion: bool = False
    policy: SolvePolicy = DEFAULT_POLICY


def change_vote(laws: RowLaws, changes: tuple[Change, ...], pad: str) -> str:
    """The exit law's vote on one update's ``changes`` (:func:`.solve_policy.vote`): a loud change keeps the world
    sweeping."""
    return vote(laws.policy, changes, flag="iteration_changed", pad=pad)


@dataclasses.dataclass(frozen=True)
class RowContext:
    """What :func:`row_update` reads of a row: its kind, impulse, metadata and response operator.

    The expressions are native code at the row index ``row``; ``kind`` is a
    local the block has already declared, and the operator reads the row's
    diagonal (:class:`.response_operator.ResponseOperator`).
    """

    row: str
    kind: str
    operator: ResponseOperator
    load: Callable[[str], str]
    store: Callable[[str, str], str]
    rhs: Callable[[str], str]
    mu: Callable[[str], str]
    weight: Callable[[str], str]
    parent: str
    ring_load: Callable[..., str]
    """The pooled normal load of the patch ring of normal row ``parent``: ``ring_load(parent)`` from the row's stored
    impulse, ``ring_load(parent, load)`` from ``load`` (a contact unit's normal in a register, :func:`contact_unit`)."""
    lanes: Lanes
    contact_kinds: tuple[int, ...] = (CONTACT, JOINT_LIMIT)
    inline_velocity_limit: bool = False
    """Velocity-limit rows the row loop visits take the velocity-limit law in place (a velocity-limit row phase)."""
    record: Callable[[str, str], str] | None = None
    """An impulse-space operator's record of the impulse a row applied (``s_app[row] += delta``)."""
    friction: bool = True
    """The rows include friction pairs; a block whose rows never do (no contacts) omits the pair's admission,
    projection, store and apply."""
    pair_law: str = "ProjectionLaw::friction_pair("
    """The friction pair's law, up to its first argument: :func:`.projection_laws.friction_pair`, or a law of the same
    arguments after leading ones of its own (a cache of the pair's metric)."""
    owned: bool = True
    """The row's first lane owns the stored impulses: the lanes synchronize before the stores (every lane has read
    the impulses the first lane overwrites; :attr:`first_store`) and, with :attr:`fenced`, after them. Otherwise every
    lane stores what it computed, as every other lane does, and reads only its own stores."""
    fenced: bool = True
    """The lanes also synchronize between the first lane's stores and the applies, which read no impulse."""
    first_store: str | None = None
    """For rows that read their impulses from registers loaded before them (a contact unit's, :func:`contact_unit`),
    the condition under which a row's stores are the first since those loads: only these wait for the lanes. ``None``:
    every row's stores wait."""
    synced: bool = True
    """The lanes synchronize after a row's stores and apply (:meth:`sync`). A contact unit's rows defer their applies
    and read no store of the unit, so its placement synchronizes once, after the unit's apply."""
    pair_residual: bool = False
    """A friction pair reduces its first tangent's residual again with the pair's terms
    (:meth:`.ResponseOperator.pair_terms` with ``own``) and solves on it in place of the row's: the free-body loop's
    friction step."""

    def sync(self) -> str:
        """The barrier after a row's stores and apply: needed when the lanes share them (the first lane's impulses, or
        an apply to the velocity or a maintained residual); nothing for one thread."""
        return self.lanes.sync() if self.owned or self.operator.applies else ""

    def store_barrier(self) -> str:
        """The barrier before a row's stores where its first lane owns them: every lane has read the impulses it
        overwrites (:attr:`first_store`); nothing otherwise."""
        barrier = self.lanes.sync() if self.owned else ""
        return f"if ({self.first_store}) {barrier}" if barrier and self.first_store else barrier


def row_update(ctx: RowContext, laws: RowLaws, *, residual: str, pad: str) -> str:
    """The projected Gauss-Seidel update of one row, after its residual's velocity term ``residual`` is reduced.

    The row's kind selects its projection (velocity limit,
    unilateral, friction pair, bilateral); a first tangent solves its pair on
    the disk of ``radius`` (declared by the block). Then one tail for every
    kind: the impulses are stored (the row's first lane owns them), an
    impulse-space operator records what it applied, and the changes are voted
    and applied, the second tangent's before the row's own.
    """
    op, row, kind = ctx.operator, ctx.row, ctx.kind
    full = residual if op.residual_includes_rhs else f"{residual} + {ctx.rhs(row)}"
    lines = [
        f"float old_impulse = {ctx.load(row)};",
        f"float residual = {full};",
        f"float row_bound = {_row_bound(ctx, op.terms(residual, ctx.rhs(row), whole=op.residual_includes_rhs))};",
    ]
    step = op.step("residual", row)
    if laws.regularize:
        lines += [
            f"float delta = {step};",
            f"if ({kind} == {CONTACT} && regularize)",
            f"    delta = ProjectionLaw::regularized_delta(delta, {ctx.weight(row)}, old_impulse);",
            "float new_impulse = ProjectionLaw::relaxed_step(old_impulse, delta, omega);",
        ]
    else:
        lines.append(f"float new_impulse = ProjectionLaw::relaxed_step(old_impulse, {step}, omega);")
    lines.append("float delta_impulse = 0.0f;")
    if ctx.friction:
        lines += ["int sib = -1;", "float sib_new = 0.0f, sib_delta = 0.0f, sib_bound = 0.0f;"]
    branches = []
    if ctx.inline_velocity_limit:
        branches.append(
            (
                f"{kind} == {VELOCITY_LIMIT}",
                f"const auto vlim = ProjectionLaw::velocity_limit_update(old_impulse, {op.scale('residual', row)});\n"
                "new_impulse = vlim.v0;\n"
                "delta_impulse = vlim.v1;",
            )
        )
    unilateral = " || ".join(f"{kind} == {k}" for k in ctx.contact_kinds)
    branches.append(
        (
            unilateral,
            "new_impulse = ProjectionLaw::unilateral(new_impulse);\ndelta_impulse = new_impulse - old_impulse;",
        )
    )
    if ctx.friction:
        pair = friction_pair(ctx, own=ctx.pair_residual)
        if _sibling_in_pair(ctx):
            # Impulses that need no barrier: the pair stores, votes and applies its second tangent in its branch, so
            # the rows without a pair carry no sibling test.
            pair += "\n" + "\n".join(
                line
                for line in (
                    ctx.store("sib", "sib_new"),
                    ctx.record("sib", "sib_delta") if ctx.record is not None else "",
                    "if (sib_delta != 0.0f) {",
                    change_vote(laws, (_sibling_change(ctx),), "    "),
                    indent(op.apply_row("sib", "sib_delta", "").lstrip("\n"), "    "),
                    "}",
                )
                if line
            )
        branches.append((f"{kind} == {FRICTION}", pair + "\ndelta_impulse = new_impulse - old_impulse;"))
    for index, (condition, body) in enumerate(branches):
        head = "if" if index == 0 else "} else if"
        lines.append(f"{head} ({condition}) {{")
        lines.append(indent(body, "    "))
    lines += ["} else {", "    delta_impulse = new_impulse - old_impulse;", "}"]
    lines += _row_tail(ctx, laws)
    return indent("\n".join(line for line in lines if line != ""), pad)


def _sibling_in_pair(ctx: RowContext) -> bool:
    """Whether a friction pair finishes its second tangent in its own branch (store, vote, apply): rows whose
    impulses need no barrier and whose operator applies the tangents one at a time."""
    return ctx.friction and not ctx.owned and not ctx.operator.fused_pair_apply


def _row_bound(ctx: RowContext, terms: str) -> str:
    """The visited row's bound (:func:`.solve_policy.bound`) from its residual's ``terms``, at its old impulse."""
    return ctx.operator.bound(ctx.rhs(ctx.row), terms, "old_impulse", ctx.row)


def _sibling_change(ctx: RowContext) -> Change:
    """The friction pair's second-tangent change as the exit law reads it (``sib_bound``: :func:`friction_pair`)."""
    return ctx.operator.change("sib_delta", "sib_bound", str(FRICTION), "sib")


def _row_tail(ctx: RowContext, laws: RowLaws) -> list[str]:
    """Store the row's (and its sibling's) impulse, record and vote the changes, apply the sibling's then the row's.

    An operator that applies a pair in one pass (``fused_pair_apply``) votes both changes together and adds the
    sibling's and then the row's change to each of its entries, written back once.
    """
    op, row, kind = ctx.operator, ctx.row, ctx.kind
    lines = [ctx.store_barrier()]
    lines.append(ctx.store(row, "new_impulse"))
    if ctx.record is not None:
        lines.append(ctx.record(row, "delta_impulse"))
    if ctx.friction and not _sibling_in_pair(ctx):
        lines += [
            "if (sib >= 0) {",
            "    " + ctx.store("sib", "sib_new"),
            ("    " + ctx.record("sib", "sib_delta")) if ctx.record is not None else "",
            "}",
        ]
    lines.append(ctx.lanes.sync() if ctx.owned and ctx.fenced else "")
    row_change = op.change("delta_impulse", "row_bound", kind, row)
    sib_change = _sibling_change(ctx)
    if ctx.friction and op.fused_pair_apply:
        # Both changes vote together, the sibling's only where the row has a pair (sib < 0: no sibling row to read).
        sib_change = dataclasses.replace(sib_change, exists="sib >= 0")
        lines += [
            "if (delta_impulse != 0.0f || sib_delta != 0.0f) {",
            change_vote(laws, (row_change, sib_change), "    "),
            indent(op.apply_pair("sib_delta", "delta_impulse", "").lstrip("\n"), "    "),
            "}",
        ]
    else:
        if ctx.friction and not _sibling_in_pair(ctx):
            lines += [
                "if (sib_delta != 0.0f) {",
                change_vote(laws, (sib_change,), "    "),
                indent(op.apply_row("sib", "sib_delta", "").lstrip("\n"), "    "),
                "}",
            ]
        lines += [
            "if (delta_impulse != 0.0f) {",
            change_vote(laws, (row_change,), "    "),
            indent(op.apply("delta_impulse", "").lstrip("\n"), "    "),
            "}",
        ]
    lines.append(ctx.sync() if ctx.synced else "")
    return lines


def pair_update(ctx: RowContext, laws: RowLaws, *, pad: str) -> str:
    """The Gauss-Seidel update of a friction pair visited as one unit, at its first tangent ``ctx.row``.

    A unit schedule solves a contact's tangent pair right after its normal
    row ``ctx.parent``, on the disk of ``radius`` (declared by the block): the
    operator reduces the first tangent's residual with the pair's terms
    (:meth:`.ResponseOperator.pair_terms`), the pair is projected as in
    :func:`row_update`, and the same tail stores, votes and applies.
    """
    lines = [
        f"const float old_impulse = {ctx.load(ctx.row)};",
        "float new_impulse = old_impulse;",
        "float delta_impulse = 0.0f;",
        "int sib = -1;",
        "float sib_new = 0.0f, sib_delta = 0.0f, row_bound = 0.0f, sib_bound = 0.0f;",
        friction_pair(ctx, own=True),
        "delta_impulse = new_impulse - old_impulse;",
        *_row_tail(ctx, laws),
    ]
    return indent("\n".join(line for line in lines if line != ""), pad)


def _sibling_bound(ctx: RowContext) -> str:
    """The friction sibling's bound from its residual (``sibling_residual``), at its old impulse."""
    op, rhs = ctx.operator, ctx.rhs("sib")
    return op.bound(rhs, op.terms("sibling_residual", rhs, whole=True), "sibling_impulse", "sib")


def friction_pair(ctx: RowContext, *, own: bool = False) -> str:
    """The friction pair at its first tangent: both tangents on the disk of ``radius`` (``sib_new``, ``sib_delta``).

    With ``own`` the first tangent's residual is reduced with the pair's terms (a pair solved as a unit).
    """
    op = ctx.operator
    lines = [
        f"sib = {ctx.parent} + 2;",
        f"const float sibling_impulse = {ctx.load('sib')};",
        op.pair_entries("sib", ""),
        "wp::vec2 pair(0.0f);",
        "// Zero load gives a zero disk; avoid unused tangent reductions.",
        "if (radius > 0.0f) {",
        op.pair_terms(ctx.row, "sib", ctx.rhs, "    ", own=own),
        f"    row_bound = {_row_bound(ctx, op.terms('residual', ctx.rhs(ctx.row), whole=True))};" if own else "",
        f"    sib_bound = {_sibling_bound(ctx)};",
        f"    pair = {ctx.pair_law}{op.value(ctx.row)}, cross, {op.value('sib')},",
        "        wp::vec2(residual, sibling_residual), wp::vec2(old_impulse, sibling_impulse), radius, omega);",
        "}",
        "pair = ProjectionLaw::disk(pair, radius);",
        "new_impulse = pair[0];",
        "sib_new = pair[1];",
        "sib_delta = sib_new - sibling_impulse;",
    ]
    return "\n".join(line for line in lines if line != "")


# ---------------------------------------------------------------------------------------------------------------------
# The row visit: a row loop's admission of row ``i`` before the row law
# ---------------------------------------------------------------------------------------------------------------------


def friction_gate(kind: str, store: str, sync: str, pad: str, *, skip: bool = True) -> str:
    """While friction is delayed (before ``friction_start_iteration``), zero a friction row's impulses and move on.

    ``store`` zeroes them (the first lane's stores, or every lane's), ``sync`` is the barrier after it (nothing for
    one thread); with ``skip`` the row loop continues.
    """
    lines = [
        f"if (ProjectionLaw::friction_gate({kind}, global_iter, friction_start_iteration)) {{",
        indent(store, "    "),
        f"    {sync}" if sync else "",
        "    continue;" if skip else "",
        "}",
    ]
    return indent("\n".join(line for line in lines if line), pad)


def admit_pair(kind: str, parent: str, pad: str, *, disk: str = "") -> str:
    """Friction rows of the row loop over ``i``: visit the first tangent only.

    Declares ``parent`` (the contact's normal row ``parent`` for a friction row, -1 for other kinds) and ``radius``,
    which :func:`friction_radius` sets, or ``disk`` (:func:`pair_radius`'s statements, indented one level) in the
    same block.
    """
    lines = [
        "int parent = -1;",
        "float radius = 0.0f;",
        f"if ({kind} == {FRICTION}) {{",
        f"    parent = {parent};",
        "    if (i != parent + 1) continue;",
        disk,
        "}",
    ]
    return indent("\n".join(line for line in lines if line), pad)


def pair_radius(
    ring_load: str, mu: str, load_row: str, load_next: str, pad: str, *, keep_next: str | None = None
) -> str:
    """Set ``radius``: ``mu`` times the pooled normal load of the contact's patch ring, and skip a zero pair.

    A zero disk with no carried tangent impulse (``load_row``, ``load_next``) cannot change the velocity, so the
    pair is skipped before its residuals are reduced. ``keep_next`` names a declared register that keeps the second
    tangent's impulse for the pair's update.
    """
    lines = [f"const float lambda_n = {ring_load};", f"radius = fmaxf({mu} * lambda_n, 0.0f);"]
    if keep_next is not None:
        lines.append(f"{keep_next} = {load_next};")
        load_next = keep_next
    lines.append(f"if (radius == 0.0f && {load_row} == 0.0f && {load_next} == 0.0f) continue;")
    return indent("\n".join(lines), pad)


def friction_radius(
    kind: str, ring_load: str, mu: str, load_row: str, load_next: str, pad: str, *, keep_next: str | None = None
) -> str:
    """:func:`pair_radius` for a friction row of the row loop (one of kind ``kind``)."""
    radius = pair_radius(ring_load, mu, load_row, load_next, pad + "    ", keep_next=keep_next)
    return f"{pad}if ({kind} == {FRICTION}) {{\n{radius}\n{pad}}}"


def row_visit(
    ctx: RowContext,
    laws: RowLaws,
    *,
    parent: str,
    check: str | None,
    declare: str = "",
    hold: str = "",
    keep_next: str | None = None,
    entries: str = "",
    residual: str | None = None,
    pad: str = "            ",
) -> str:
    """Visit row ``i`` (``ctx.row``) of a row loop, its kind ``ctx.kind`` declared: admission, residual and row law.

    A friction row waits for ``friction_start_iteration`` with a zero impulse (:func:`friction_gate`); a pair is
    visited at its first tangent (``parent``: the row's parent, :func:`admit_pair`) and skipped while its disk and
    impulses are zero (:func:`friction_radius`), before its residual is reduced. Every other row needs a positive
    diagonal ``check`` (its value or stored inverse, declared by ``declare``), unless the schedule admitted the rows
    on it already (``None``). ``hold`` declares, after the check, registers the disk and the law read (``keep_next``
    keeps the second tangent's impulse in one); the disk joins the admission's block when nothing comes between
    them. ``entries`` declares what the operator reads of the row and ``residual`` reduces ``jv`` (the operator's
    :meth:`~.ResponseOperator.residual` by default). Rows without friction pairs (``ctx.friction`` false) have no
    gate, admission or disk. A skipped row continues the loop.
    """
    row, kind, op = ctx.row, ctx.kind, ctx.operator
    pairs = ctx.friction
    between = [indent(declare, pad) if declare else ""]
    if check is not None:
        exempt = f" && {kind} != {FRICTION}" if pairs else ""
        between.append(f"{pad}if ({check} <= 0.0f{exempt}) continue;")
    between.append(indent(hold, pad) if hold else "")
    lines = [friction_gate(kind, ctx.store(row, "0.0f"), ctx.sync(), pad) if pairs else ""]
    disk = (ctx.ring_load("parent"), ctx.mu(row), ctx.load(row), ctx.load(f"{row} + 1"))
    if pairs and not any(between):
        lines.append(admit_pair(kind, parent, pad, disk=pair_radius(*disk, "    ", keep_next=keep_next)))
    else:
        lines += [
            admit_pair(kind, parent, pad) if pairs else "",
            *between,
            friction_radius(kind, *disk, pad, keep_next=keep_next) if pairs else "",
        ]
    lines += [
        indent(entries, pad) if entries else "",
        op.residual("jv", pad) if residual is None else residual,
        row_update(ctx, laws, residual="jv", pad=pad),
    ]
    return "\n".join(line for line in lines if line)


# ---------------------------------------------------------------------------------------------------------------------
# The contact unit: a contact normal and its friction pair on the unit's Gram block
# ---------------------------------------------------------------------------------------------------------------------


class UnitOperator:
    """A placement operator's rows in a contact unit (:func:`contact_unit`): a mixin ahead of the operator.

    A contact unit is a contact normal (the unit's row ``r == 0``) and its friction pair (``r == 1``: the first tangent,
    the second its sibling), visited by the row law (:func:`row_visit`) one after the other on the unit's Gram block
    ``G = Z^T Z``. The placement reduces the unit's three velocity terms once, before its rows (``unit_jv0``,
    ``unit_jv1``, ``unit_jv2``), and gives the pair's Gram couplings with the normal (``gram_n1``, ``gram_n2``): a cache
    formed once per launch, the builder's record, or the same reduction. The rows read the pair's terms moved by the
    normal's change, ``Z_t^T (u + Z_n dn) = Z_t^T u + G_tn dn``, and defer their applies into ``unit_dn``, ``unit_dt1``
    and ``unit_dt2``, which the placement applies once after the unit: in exact arithmetic the row-by-row sweep. The
    pair's coupling ``Z_t1 . Z_t2`` is the placement's (:meth:`coupling`), or none where its pair law reads a metric
    cached for the unit (:class:`CachedPairMetric`).

    A row's vote reads its terms' magnitude through a lower bound (:meth:`terms`): the reduced term's own, or, where
    the placement's ladder also reduced the magnitudes (:attr:`magnitudes`: ``unit_jv{k}_terms``, and ``C_t = sum_k
    |Z_tk Z_nk|`` in ``gram_n{t}_terms``), the normal's ``S_n`` and a moved tangent's ``max(|Z_t^T u'|, S_t - |dn|
    C_t)``, the row-by-row sweep's ``S_t`` itself when the normal did not move.
    """

    magnitudes = False

    def terms(self, out: str, rhs: str, *, whole: bool) -> str:
        if not self.magnitudes:
            return super().terms(out, rhs, whole=whole)
        moved = terms_bound(out, rhs) if whole else f"fabsf({out})"
        if out == "sibling_residual":
            return f"fmaxf({moved}, unit_jv2_terms - fabsf(unit_dn) * gram_n2_terms)"
        return f"(r == 0 ? unit_jv0_terms : fmaxf({moved}, unit_jv1_terms - fabsf(unit_dn) * gram_n1_terms))"

    def residual(self, out: str, indent: str) -> str:
        return f"{indent}float {out} = r == 0 ? unit_jv0 : unit_jv1 + gram_n1 * unit_dn;"

    def sibling_residual(self, out: str, sibling: str, indent: str, rhs: str) -> str:
        return f"{indent}float {out} = (unit_jv2 + gram_n2 * unit_dn) + {rhs};"

    def pair_entries(self, sibling: str, indent: str) -> str:
        return ""

    def pair_terms(self, row: str, sibling: str, rhs: Callable[[str], str], indent: str, *, own: bool = False) -> str:
        if own:
            raise ValueError("a contact unit's first tangent reads its moved residual through the row visit")
        lines = (
            self.sibling_residual("sibling_residual", sibling, indent, rhs(sibling)),
            self.coupling("cross", row, sibling, indent),
        )
        return "\n".join(line for line in lines if line != "")

    def apply(self, delta: str, indent: str) -> str:
        return f"\n{indent}if (r == 0) unit_dn += {delta}; else unit_dt1 += {delta};"

    def apply_row(self, row: str, delta: str, indent: str) -> str:
        return f"\n{indent}unit_dt2 += {delta};"

    def apply_pair(self, sibling_delta: str, delta: str, indent: str) -> str:
        return f"\n{indent}unit_dt2 += {sibling_delta};" + self.apply(delta, indent)


class CachedPairMetric:
    """A unit operator mixin whose pair law reads the pair's metric cached for the unit (:func:`cached_pair_law`): the
    visit forms no coupling."""

    def coupling(self, out: str, row: str, sibling: str, indent: str) -> str:
        return f"{indent}const float {out} = 0.0f;  // the cached metric replaces the coupling"


def cached_pair_law(name: str, scale: str, low: str, high: str, axis: tuple[str, str], pad: str) -> str:
    """Declare ``name``, the friction pair's law (:attr:`RowContext.pair_law`'s arguments after one leading ``int``)
    from a metric cached for the unit: its tangent ``scale``, the two eigenvalues ``low`` and ``high`` and the larger's
    ``axis`` (:func:`.projection_laws.friction_pair_from_metric`); the visit's diagonals and coupling go unread."""
    lines = [
        f"const auto {name} = [&](int, float, float, float, wp::vec2 residual, wp::vec2 old,",
        f"{' ' * (len(name) + 18)}float radius, float relaxation) -> wp::vec2 {{",
        f"    return ProjectionLaw::friction_pair_from_metric({scale}, {low}, {high},",
        f"        wp::vec2({axis[0]}, {axis[1]}), residual, old, radius, relaxation);",
        "};",
    ]
    return "\n".join(pad + line for line in lines)


def contact_unit(
    ctx: RowContext,
    laws: RowLaws,
    *,
    rows: str,
    declare: str,
    parent: str,
    check: str | None,
    held: tuple[str, str, str] | None = None,
    pad: str,
) -> str:
    """The rows of a contact unit by the row law (BACKBONE §1 step 5), after the placement's one reduction of the
    unit's terms and before its one apply of the unit's changes (both the placement's).

    Row ``r`` of the unit (the normal, then its pair at the first tangent; ``rows`` is the unit's row count, one for a
    normal without a pair, else three) declares its index ``i``, kind and response (``declare``, which ``check`` reads:
    the row law runs when it is positive; ``None`` where the placement admitted the unit's normal) and runs
    :func:`row_visit` with the unit's operator (:class:`UnitOperator`), the pair at the normal ``parent``. A visit that
    skips its row leaves it (a one-pass loop), not the unit.

    The rows defer their applies, so none synchronizes the lanes after it: the placement does once, after the unit's
    apply. Where the rows' first lane owns the stores, the placement holds the unit's three impulses in registers
    (``held``: the normal's and the two tangents', loaded before its reduction), which the rows read, and the pair
    pools its ring from the normal's new impulse in a register (``unit_normal``, what the normal stores). The rows then
    read no store of the unit, so only the unit's first stores wait for every lane's reads: the normal's where the
    placement admitted it, else each row's (a skipped normal leaves the pair's stores first). Rows that every lane
    stores alike, or one thread's, read only their own stores.
    """
    if (ctx.owned and ctx.lanes.parallel) != (held is not None):
        raise ValueError("a contact unit holds its impulses in registers exactly where its first lane owns the stores")
    unit = dataclasses.replace(ctx, fenced=False, synced=False, first_store="r == 0" if check is None else None)
    declarations = [f"{pad}float unit_dn = 0.0f, unit_dt1 = 0.0f, unit_dt2 = 0.0f;"]
    if held is not None:
        normal, tangent, sibling = held
        loads = {ctx.row: f"(r == 0 ? {normal} : {tangent})", f"{ctx.row} + 1": sibling, "sib": sibling}

        def store(row: str, value: str) -> str:
            stored = ctx.store(row, value)
            return f"{stored}\nif (r == 0) unit_normal = {value};" if row == ctx.row else stored

        unit = dataclasses.replace(
            unit,
            load=lambda row: loads.get(row) or ctx.load(row),
            store=store,
            ring_load=lambda row: ctx.ring_load(row, "unit_normal"),
        )
        declarations.append(f"{pad}float unit_normal = {normal};")
    visit = row_visit(unit, laws, parent=parent, check=check, pad=pad + "    ")
    lines = [
        *declarations,
        "#pragma unroll",
        f"{pad}for (int r = 0; r < 2; ++r) {{",
        f"{pad}    if (r >= {rows}) break;",
        f"{pad}    // One pass of the row law; its `continue` leaves the row.",
        f"{pad}    for (int once = 0; once < 1; ++once) {{",
        indent(declare, pad + "        "),
        visit,
        f"{pad}    }}",
        f"{pad}}}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------------------------------
# The exit: the end of every sweep
# ---------------------------------------------------------------------------------------------------------------------

QUIET = "BALLOT(iteration_changed != 0) == 0u"
"""No lane of the sweep's group voted a loud change."""

BLOCK_QUIET = "!ANY(iteration_changed != 0)"
"""No lane of a block group of several warps voted a loud change: the block's vote, a barrier."""


def end_sweep(
    policy: SolvePolicy,
    *,
    iteration: str,
    gated: bool = True,
    violation: tuple[str, str, str, str] | None = None,
    block: bool = False,
    iterations: str = "iterations",
    argmax: str | None = "ARGMAX",
) -> str:
    """The end of a sweep under the exit law (:func:`.solve_policy.sweep_exit`): stop once no update was loud.

    A ``gated`` sweep stops only once friction is active (delayed friction is a change to come). ``violation`` bounds
    the rows a sweep of selected rows leaves unvisited. A ``block`` group spans several warps: its quiet test is the
    block's vote (:data:`BLOCK_QUIET`) and ``argmax`` its trace reduction (:func:`.solve_policy.block_argmax`).
    ``iterations`` is the sweep budget.
    """
    gate = "global_iter >= friction_start_iteration" if gated else None
    quiet = BLOCK_QUIET if block else QUIET
    return sweep_exit(
        policy,
        gate=gate,
        quiet=quiet,
        violation=violation,
        iteration=iteration,
        iterations=iterations,
        argmax=argmax,
    )


# ---------------------------------------------------------------------------------------------------------------------
# Row blocks: a row store, an operator and the row law, over a range of rows in slot order
# ---------------------------------------------------------------------------------------------------------------------


def dense_rows_block(
    rows: WorldRows,
    op,
    laws: RowLaws,
    *,
    admission: str = "",
    pad: str = "        ",
) -> str:
    """The Gauss-Seidel loop over a world's dense rows ``[0, m_dense)`` in slot order, on packed rows
    (:class:`.response_operator.SparseFactor`).

    ``admission`` holds the block's own skips, run after the row's type is read.
    A friction pair's second tangent shares its
    first tangent's entries: the first tangent's residual reduces the pair's
    terms with its own (:meth:`.SparseFactor.residual`).
    """
    inner = pad + "    "
    torsion_skip = f"if (row_type == {TORSION}) continue;" if laws.torsion else ""
    op = dataclasses.replace(op, diagonal=lambda row: "denom" if row == "i" else rows.diag(row))
    ctx = RowContext(
        row="i",
        kind="row_type",
        operator=op,
        load=rows.load,
        store=rows.store,
        rhs=rows.rhs_of,
        mu=rows.mu,
        weight=rows.weight,
        parent="parent",
        ring_load=rows.ring_load,
        lanes=rows.lanes,
    )
    residual = op.residual("jv", inner, sibling="parent + 2", paired=f"row_type == {FRICTION}")
    body = [
        f"{inner}int row_type = {rows.kind('i')};",
        indent(torsion_skip, inner) if torsion_skip else "",
        indent(admission, inner) if admission else "",
        row_visit(
            ctx,
            laws,
            parent=rows.parent("i"),
            check="denom",
            declare=f"float denom = {rows.diag('i')};",
            residual=residual,
            pad=inner,
        ),
    ]
    return f"{pad}for (int i = 0; i < m_dense; i++) {{\n" + "\n".join(line for line in body if line) + f"\n{pad}}}"


def free_body_rows_block(
    rows: FreeBodyRows,
    op,
    laws: RowLaws,
    *,
    admission: str = "",
    end: str = "mf_contact_end",
    pad: str = "        ",
) -> str:
    """The Gauss-Seidel loop over a world's free-body rows ``[0, end)`` in slot order.

    The impulses need no barrier around their stores: every lane stores the value it computed (or, streamed, the
    first lane stores and every read takes the first lane's value), so a lane only ever reads its own stores.
    """
    inner = pad + "    "
    op = dataclasses.replace(op, inverse=lambda row: "mf_diag" if row == "i" else rows.inverse_meta(row))
    ctx = RowContext(
        row="i",
        kind="mf_rt",
        operator=op,
        load=rows.load,
        store=rows.store,
        rhs=lambda row: "__int_as_float(meta.z)" if row == "i" else rows.rhs_meta(row),
        mu=rows.mu,
        weight=rows.weight,
        parent="parent",
        ring_load=rows.ring_load,
        lanes=rows.lanes,
        contact_kinds=(CONTACT,),
        owned=False,
    )
    body = [
        op.consume("i", end, inner),
        indent(admission, inner) if admission else "",
        row_visit(ctx, laws, parent=rows.parent_of("packed_tp"), check="mf_diag", pad=inner),
    ]
    return (
        f"{op.pipeline_registers(pad)}\n"
        f"{pad}if ({end} > 0) {{\n{op.prefetch_first('0', pad + '    ')}\n{pad}}}\n"
        f"{pad}for (int i = 0; i < {end}; i++) {{\n" + "\n".join(line for line in body if line) + f"\n{pad}}}"
    )


# ---------------------------------------------------------------------------------------------------------------------
# End-of-sweep passes: rows whose law gives them the last word of every sweep
# ---------------------------------------------------------------------------------------------------------------------


def velocity_limit_pass(rows: WorldRows, op, laws: RowLaws, *, end: str = "m_dense", pad: str = "            ") -> str:
    """Dense joint velocity-limit rows after every other row of the sweep, under the velocity-limit law.

    Each row keeps a unilateral total (:func:`.projection_laws.velocity_limit_update`), owned by the row's first
    lane. ``rows`` stages the rows' impulses, kinds, diagonals and right-hand sides (:class:`WorldRows`, or a route's
    store of the same accessors); the operator reduces each row's velocity term outside the row loop
    (:meth:`~.ResponseOperator.row_velocity`).
    """
    lanes = rows.lanes
    lines = [
        f"for (int i = 0; i < {end}; i++) {{",
        f"    if (({rows.kind('i', scan=True)}) != {VELOCITY_LIMIT}) continue;",
        f"    float denom = {rows.diag('i')};",
        "    if (denom <= 0.0f) continue;",
        "",
        op.row_velocity("my_sum_vlim", "i", "    "),
        "    float jv_vlim = my_sum_vlim;",
        "",
        f"    float residual_vlim = jv_vlim + {rows.rhs_of('i')};",
        f"    const float old_vlim = {rows.load('i')};",
        f"    const float bound_vlim = {bound(rows.rhs_of('i'), op.terms('my_sum_vlim', '', whole=False), 'old_vlim', diagonal='denom')};",
        "    const auto vlim = ProjectionLaw::velocity_limit_update(old_vlim, residual_vlim / denom);",
        "    const float delta_impulse_vlim = vlim.v1;",
        f"    {lanes.sync()}",
    ]
    change = Change("delta_impulse_vlim", "bound_vlim", str(VELOCITY_LIMIT), "denom", scaled=True)
    lines += [
        f"    {rows.store('i', 'vlim.v0')}",
        "",
        "    if (delta_impulse_vlim != 0.0f) {",
        change_vote(laws, (change,), "        "),
        op.apply_strided("i", "delta_impulse_vlim", "        "),
        "    }",
        f"    {lanes.sync()}",
        "}",
    ]
    return indent("\n".join(lines), pad)


def speed_rows_pass(rows: FreeBodyRows, op, laws: RowLaws, *, pad: str = "            ") -> str:
    """Free-body speed rows (``[mf_contact_end, m_mf)``) after every other row: each bounds the magnitude of its
    body's linear (axis ``e_0``) or angular (axis ``e_3``) velocity, along that velocity (:func:`speed_row`)."""
    lanes = rows.lanes
    lines = [
        "for (int i = mf_contact_end; i < m_mf; i++) {",
        "    int4 meta = *reinterpret_cast<const int4*>(&mf_meta.data[off_meta + i * 4]);",
        "    int packed_tp = meta.w;",
        "    int mf_rt = packed_tp & 0xFFFF;",
        f"    if (mf_rt != {VELOCITY_LIMIT}) continue;",
        "",
        "    int packed_dofs = meta.x;",
        "    int dof_a = packed_dofs >> 16;",
        "    float mf_diag = __int_as_float(meta.y);",
        "    if (mf_diag <= 0.0f || dof_a < 0) continue;",
        "",
        f"    const int speed_axis = {op.Z_a}.data[{op.base} + i * 6 + 3] != 0.0f ? 3 : 0;",
        speed_row(
            op,
            laws,
            load=rows.load("i"),
            store=lanes.first(rows.store("i", "speed_row[0]")) if not rows.stream else rows.store("i", "speed_row[0]"),
            lanes=lanes,
            rhs="__int_as_float(meta.z)",
            inverse="mf_diag",
            pad="    ",
        ),
        "}",
    ]
    return indent("\n".join(lines), pad)


def speed_row(
    op,
    laws: RowLaws,
    *,
    load: str,
    store: str,
    lanes: Lanes,
    rhs: str,
    inverse: str,
    body: str = "dof_a",
    axis: str = "speed_axis",
    pad: str = "    ",
) -> str:
    """The update of a free-body speed row on its body's velocity coordinates ``[axis, axis + 3)``.

    The body's coordinates from ``body`` hold its factor coordinates; the speed is the magnitude of its physical
    velocity there (:meth:`.FreeBodyZ.speed`). The row's law (:func:`.projection_laws.speed_row`) gives its stored
    value (``store`` stores ``speed_row[0]``, the total of the row's impulse ``load``) and the scale that
    pushes the velocity back along itself (:meth:`.FreeBodyZ.scale_speed`). ``inverse`` is the row's stored inverse
    diagonal.
    """
    lines = [
        op.speed("speed_mf_vlim", body, axis, pad),
        f"{pad}const float old_mf_vlim = {load};",
        f"{pad}const float bound_speed = {bound(rhs, 'speed_mf_vlim', 'old_mf_vlim', step=inverse)};",
        f"{pad}const wp::vec3 speed_row = ProjectionLaw::speed_row(speed_mf_vlim, {rhs}, {inverse}, old_mf_vlim);",
    ]
    if lanes.sync():
        lines.append(f"{pad}{lanes.sync()}")
    lines += [
        f"{pad}{store}",
        "",
        f"{pad}if (speed_row[2] != 0.0f) {{",
        change_vote(
            laws,
            (Change("speed_row[1]", "bound_speed", str(VELOCITY_LIMIT), f"1.0f / {inverse}"),),
            pad + "    ",
        ),
        op.scale_speed("speed_row[2]", body, axis, pad + "    "),
        f"{pad}}}",
    ]
    if lanes.sync():
        lines.append(f"{pad}{lanes.sync()}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------------------------------
# The dense-row tiers' worlds: a block per world without free-body rows, its dense rows within the tier's bounds
# ---------------------------------------------------------------------------------------------------------------------


def dense_tier_world(
    capacity: int,
    min_rows: int,
    dofs: int,
    max_constraints: int,
    *,
    rows: str,
    support: int,
    pad: str = "    ",
) -> str:
    """Enter a world of a dense-row tier, or return: one without free-body rows and with more than ``min_rows`` and
    at most ``capacity`` dense rows.

    Declares ``m_dense``, ``m_mf``, ``rows`` (the rows the tier holds, ``m_dense``), ``dof_map_base``,
    ``off_dense``, and the world's first packed row entry ``packed_base`` (rows of ``support`` entries:
    :mod:`.packed_rows`).
    """
    base = f"const size_t packed_base = (size_t)world * {max_constraints * int(support)};"
    return f"""{pad}int m_dense = world_constraint_count.data[world];
{pad}const int m_mf = mf_constraint_count.data[world];
{pad}if (m_dense == 0 || m_mf != 0) return;
{pad}if (m_dense > {capacity} || m_dense <= {min_rows}) return;

{pad}const int {rows} = m_dense;
{pad}const int dof_map_base = world * {dofs};
{pad}const int off_dense = world * {max_constraints};
{pad}{base}"""


# ---------------------------------------------------------------------------------------------------------------------
# The Jacobi law: every row of a world at once, from the extrapolated impulses, with momentum
# ---------------------------------------------------------------------------------------------------------------------

JACOBI_UNILATERAL, JACOBI_FRICTION, JACOBI_BILATERAL, JACOBI_FROZEN = 0, 1, 2, -1
"""The kinds of a Jacobi row (``s_kind``): its projection, decoded from its type once at staging. A unilateral row
(contact normals, position limits, velocity limits) clamps at zero, a friction row's pair projects on its
disk and a bilateral row takes its step; a frozen row keeps its warm start."""


@dataclasses.dataclass(frozen=True)
class JacobiRows:
    """A world's dense rows ``[0, rows)`` under the Jacobi law, a lane each (row ``i`` on lane ``i``, ``threads``
    lanes, at most ``capacity`` rows).

    Staged per row: its impulse ``s_x``; the extrapolated impulse the next step
    reads, ``s_y``; its warm start ``s_lam0``; its right-hand side ``s_rhs``
    (once the operator has staged, the residual at zero impulse); its scaled
    step ``s_step``; its friction coefficient and parent; and its kind
    ``s_kind`` (:data:`JACOBI_UNILATERAL` and the others). ``s_x`` and ``s_y``
    are float4-aligned and padded with zeros to a multiple of four rows. With
    ``limits``, which rows are joint velocity limits are staged (``s_vlim``, for
    the last word: :func:`jacobi_limit_pass`). Every row is admitted
    provisionally: the operator forms its diagonal from the
    staged rows (whitened in the kernel), and a row whose formed diagonal is
    not positive is frozen once every lane has read the staged kinds (its
    lane's ``frozen``). The lane keeps its row's diagonal in a register
    (``row_diag``), which the exit law scales the row's impulse by; with
    ``types`` each row's constraint type is staged (``s_type``), which the
    trace names its change by.
    """

    capacity: int
    threads: int
    limits: bool = False
    rows: str = "n_rows"
    types: bool = False

    def buffers(self) -> tuple[Buffer, ...]:
        """The rows' state and metadata."""
        AM = self.capacity
        state = (Buffer("s_x", "float", AM, align=16), Buffer("s_y", "float", AM, align=16))
        floats = (Buffer(name, "float", AM) for name in ("s_lam0", "s_rhs", "s_step", "s_mu"))
        ints = ("s_parent", "s_kind", *(("s_vlim",) if self.limits else ()), *(("s_type",) if self.types else ()))
        return (*state, *floats, *(Buffer(name, "int", AM) for name in ints))

    def loads(
        self,
        *,
        rhs: str,
        row_type: str,
        parent: str,
        mu: str,
        unilateral: tuple[int, ...],
        pad: str = "    ",
    ) -> str:
        """Stage each row, lane-strided: its impulses at their warm start, its metadata (native expressions at row
        ``i``) and its kind, its diagonal formed later (:func:`jacobi_scale`). ``unilateral`` lists the row types that
        clamp at zero."""
        decode = [
            f"if ({' || '.join(f'row_type == {kind}' for kind in unilateral)}) kind = {JACOBI_UNILATERAL};",
            f"else if (row_type == {FRICTION}) kind = {JACOBI_FRICTION};",
        ]
        lines = [
            f"for (int i = lane; i < {self.rows}; i += {self.threads}) {{",
            "    const float l0 = world_impulses.data[off_dense + i];",
            "    s_lam0[i] = l0; s_x[i] = l0; s_y[i] = l0;",
            f"    s_rhs[i] = {rhs};",
            f"    const int row_type = {row_type} & {_DENSE_META_ROW_TYPE_MASK};",
            f"    s_parent[i] = {parent};",
            f"    s_mu[i] = {mu};",
            f"    s_vlim[i] = row_type == {VELOCITY_LIMIT} ? 1 : 0;" if self.limits else "",
            f"    int kind = {JACOBI_BILATERAL};",
            *("    " + line for line in decode),
            "    s_kind[i] = kind;",
            "    s_step[i] = 1.0f;",
            "    s_type[i] = row_type;" if self.types else "",
            "}",
        ]
        lines.insert(0, "int frozen = 0;")
        # The lane's own row's diagonal (a row each: the loop runs at most once per lane).
        lines.insert(0, "float row_diag = 0.0f;")
        return indent("\n".join(line for line in lines if line), pad)

    def freeze(self, pad: str = "    ") -> str:
        """Freeze the rows whose formed diagonal is not positive, after the staging's barrier; synchronized."""
        return f"{pad}if (frozen) s_kind[lane] = {JACOBI_FROZEN};\n{pad}SYNC();"

    def padding(self) -> str:
        """Zero the impulses past the rows, up to a multiple of four rows (the float4 reads of ``s_y``)."""
        rows = self.rows
        return (
            f"for (int j = {rows} + lane; j < (({rows} + 3) & ~3); j += {self.threads}) "
            "{ s_y[j] = 0.0f; s_x[j] = 0.0f; }"
        )

    def ring_load(self, parent: str) -> str:
        """The pooled normal load of the patch ring of normal row ``parent``; a parent past the rows ends the ring."""
        rows = self.rows
        return (
            f"RING_LOAD(s_x[{parent}], {parent}, {rows},\n"
            f"    [&](int r) {{ const int next = s_parent[r]; return next < {rows} ? next : -1; }},\n"
            "    [&](int r) { return s_x[r]; })"
        )

    def stores(self, pad: str = "    ") -> str:
        """Write the rows' impulses back."""
        return (
            f"{pad}for (int i = lane; i < {self.rows}; i += {self.threads}) {{\n"
            f"{pad}    world_impulses.data[off_dense + i] = s_x[i];\n{pad}}}"
        )


def jacobi_scale(aii: str, bound: str, *, cfm: str, pad: str) -> str:
    """Set admitted row ``i``'s Jacobi step: ``omega w_i / (A_ii + cfm)`` with ``w_i = min(A_ii / bound, 1)`` (zero
    for a zero bound), ``bound`` a bound of the row's absolute sum ``sum_j |A_ij|``.

    ``w_i`` is the row's share of the response of the rows it couples to (generalized mass splitting), which keeps
    the simultaneous steps from diverging where many rows couple through one body. The diagonal ``A_ii + cfm`` (the
    row's compliance) is formed here and a row without a positive one is marked for freezing (``frozen``,
    :meth:`JacobiRows.freeze`: the other lanes are still reading the kinds). The formed diagonal is the lane's
    ``row_diag``, which the exit law reads.
    """
    return indent(
        "\n".join(
            [
                f"const float w = ({bound} > 0.0f) ? fminf({aii} / {bound}, 1.0f) : 0.0f;",
                f"const float diag_i = {aii} + {cfm};",
                "row_diag = diag_i;",
                "if (diag_i > 0.0f) s_step[i] = w * omega / diag_i; else { s_step[i] = 0.0f; frozen = 1; }",
            ]
        ),
        pad,
    )


def isotropic_pair_step(rows: JacobiRows, pad: str = "    ") -> str:
    """Give both tangents of every friction pair the smaller of their steps, then synchronize.

    The radial projection onto the friction disk is then the projection in the pair's own metric, so a sliding pair
    converges to maximal dissipation (the Gauss-Seidel pair solve's friction law). A row keeps its own step when its
    sibling is not an admitted friction row of the same normal, or when either step is zero (a zero-Jacobian tangent
    would otherwise freeze its live sibling). The pair's first tangent's lane handles it.
    """
    F = JACOBI_FRICTION
    lines = [
        f"for (int i = lane; i < {rows.rows}; i += {rows.threads}) {{",
        f"    if (s_kind[i] == {F} && i == s_parent[i] + 1) {{",
        "        const int j = i + 1;",
        f"        if (j < {rows.rows} && s_kind[j] == {F} && s_parent[j] == s_parent[i]) {{",
        "            const float step_a = s_step[i];",
        "            const float step_b = s_step[j];",
        "            if (step_a > 0.0f && step_b > 0.0f) {",
        "                const float step_pair = fminf(step_a, step_b);",
        "                s_step[i] = step_pair;",
        "                s_step[j] = step_pair;",
        "            }",
        "        }",
        "    }",
        "}",
        "SYNC();",
    ]
    return indent("\n".join(lines), pad)


def _jacobi_step(rows: JacobiRows, op, pad: str) -> str:
    """Row ``i``'s update ``x_new`` from the extrapolated impulses (``x_old``: its impulse), on its lane."""
    update = [
        "value = s_y[i] - s_step[i] * r;",
        f"if (s_kind[i] == {JACOBI_UNILATERAL}) value = ProjectionLaw::unilateral(value);",
        f"if (s_kind[i] == {JACOBI_FRICTION} && global_iter < friction_start_iteration) value = 0.0f;",
    ]
    lines = [
        "float x_new = 0.0f, x_old = 0.0f, x_from = 0.0f, row_bound = 0.0f;",
        "{",
        "    const int i = lane;",
        f"    if (i < {rows.rows}) {{",
        "        const float xi = s_x[i];",
        "        x_old = xi;",
        "        x_from = xi;",
        "        float value = xi;",
        "        if (s_kind[i] >= 0) {",
        op.jacobi_residual("r", "            "),
        f"            row_bound = {bound('row_rhs', op.terms('r', 'row_rhs', whole=True), 'row_share * xi', step='row_step')};",
        "            x_from = s_y[i];",
        *("            " + line for line in update),
        "        }",
        "        x_new = value;",
        "    }",
        "}",
    ]
    return indent("\n".join(lines), pad)


def _jacobi_projection(rows: JacobiRows, *, pooled: bool, pad: str) -> str:
    """Project each friction row's update radially onto its pair's disk of radius ``mu`` times the normal load.

    The load is the pair's normal's fresh impulse (``pooled``: the sum over its patch ring of normals, in ring order
    from its own). One warp exchanges the fresh updates by shuffle (a pooled load reads the ring's staged ones, and
    the warp synchronizes before the impulses are stored again); several warps stage them in ``s_x``. The pair's
    tangents project together, each on its own lane.
    """
    F, n = JACOBI_FRICTION, rows.rows
    disk = "radius <= 0.0f ? 0.0f : ProjectionLaw::disk(wp::vec2({first}, {second}), radius)[0]"
    if rows.threads == 32:
        load = "xp"
        lines = [
            "{",
            "    const int i = lane;",
            f"    const bool fr = (i < {n}) && (s_kind[i] == {F});",
            "    const int p = fr ? s_parent[i] : i;",
            "    const int sib = fr ? ((i == p + 1) ? p + 2 : p + 1) : i;",
        ]
        if pooled:
            lines += [
                "    const float xs = READ(x_new, sib & 31);",
                f"    if (i < {n}) s_x[i] = x_new;",
                "    WARP_SYNC();",
                "    if (fr) {",
                indent(f"const float load = {rows.ring_load('p')};", "        "),
            ]
            load = "load"
        else:
            lines += [
                "    const float xp = READ(x_new, p & 31);",
                "    const float xs = READ(x_new, sib & 31);",
                "    if (fr) {",
            ]
        lines += [
            f"        const float radius = fmaxf(s_mu[i] * {load}, 0.0f);",
            f"        x_new = {disk.format(first='x_new', second='xs')};",
            "    }",
            "    WARP_SYNC();" if pooled else "",
            "}",
        ]
        return indent("\n".join(line for line in lines if line), pad)
    load = "s_x[p]"
    ring = []
    if pooled:
        ring = [indent(f"const float load = {rows.ring_load('p')};", "        ")]
        load = "load"
    lines = [
        "SYNC();",
        f"if (lane < {n}) s_x[lane] = x_new;",
        "SYNC();",
        "{",
        "    const int i = lane;",
        f"    if (i < {n} && s_kind[i] == {F}) {{",
        "        const int p = s_parent[i];",
        "        const int sib = (i == p + 1) ? p + 2 : p + 1;",
        *ring,
        f"        const float radius = fmaxf(s_mu[i] * {load}, 0.0f);",
        f"        x_new = {disk.format(first='s_x[i]', second='s_x[sib]')};",
        "    }",
        "}",
        "SYNC();",
    ]
    return indent("\n".join(lines), pad)


def _jacobi_momentum(rows: JacobiRows, *, pad: str) -> str:
    """Nesterov's extrapolation factor ``beta`` with adaptive restart: the momentum restarts (``t_k = 1``) when the
    last step points against it, ``(y - x_new) . (x_new - x_old) > 0`` summed over the rows (per warp, then over the
    warps in order)."""
    lines = [
        "float dot_local = 0.0f;",
        f"if (lane < {rows.rows}) dot_local = (s_y[lane] - x_new) * (x_new - x_old);",
        "float dot = dot_local;",
        "SUM(dot);",
    ]
    if rows.threads > 32:
        lines += [
            "if ((lane & 31) == 0) s_dot[lane >> 5] = dot;",
            "SYNC();",
            "dot = 0.0f;",
            "#pragma unroll",
            f"for (int wv = 0; wv < {rows.threads // 32}; ++wv) dot += s_dot[wv];",
        ]
    lines += [
        "if (dot > 0.0f) t_k = 1.0f;",
        "const float t_next = 0.5f * (1.0f + sqrtf(1.0f + 4.0f * t_k * t_k));",
        "const float beta = (t_k - 1.0f) / t_next;",
        "t_k = t_next;",
    ]
    if rows.threads > 32:
        lines.append("SYNC();")
    return indent("\n".join(lines), pad)


def jacobi_sweeps(rows: JacobiRows, op, laws: RowLaws, *, sweeps: int, pooled: bool, pad: str = "    ") -> str:
    """The Jacobi sweeps of a world's rows over the response operator ``op`` (its Jacobi actions), at most ``sweeps``.

    Both tangents of a friction pair first take one step (:func:`isotropic_pair_step`). Each sweep, every admitted
    row steps at once from the extrapolated impulses ``y``: its residual ``r = b' + (A y)_i``
    (:meth:`~.ResponseOperator.jacobi_residual`, after the operator's :meth:`~.ResponseOperator.gather`), then
    ``y_i - s_step_i r`` clamped at zero for a unilateral row, or zero for a friction row while friction is delayed.
    Each friction row is then projected radially onto
    its pair's disk (``pooled``: on its patch ring's pooled normal load), the impulses advance with Nesterov's
    momentum and adaptive restart, each row votes its change under the exit law (:func:`change_vote`:
    its step from the extrapolated impulse over its step size, which is the row's residual there for an unclamped row,
    against its floor; the trace names it by the row's staged type, :attr:`JacobiRows.types`), and the sweep ends
    under the policy (:func:`end_sweep`, the budget ``sweeps``; a block of several warps reduces the trace with
    ``BLOCK_ARGMAX``, :func:`.solve_policy.block_argmax`).
    """
    policy = laws.policy
    if policy.traces and not rows.types:
        raise ValueError("the trace names the Jacobi rows' changes by their staged types")
    inner = pad + "    "
    change = Change("x_new - x_from", "row_bound", "s_type[i]" if rows.types else "-1", "row_inv_step")
    block = rows.threads > 32
    lines = [
        rows.freeze(pad),
        isotropic_pair_step(rows, pad),
        # The lane's row: its residual at zero impulse and its step size, fixed through the sweeps.
        f"{pad}const float row_rhs = lane < {rows.rows} ? s_rhs[lane] : 0.0f;",
        f"{pad}const float row_step = lane < {rows.rows} ? s_step[lane] : 0.0f;",
        f"{pad}const float row_inv_step = row_step > 0.0f ? 1.0f / row_step : 0.0f;",
        # The step's share of a Gauss-Seidel step (omega w_i): the bound's impulse term in the step's units.
        f"{pad}const float row_share = row_diag * row_step;",
        f"{pad}float t_k = 1.0f;",
        f"{pad}for (int sweep = 0; sweep < {sweeps}; ++sweep) {{",
        f"{inner}const int global_iter = iteration_offset + sweep;",
        op.gather(inner),
        _jacobi_step(rows, op, inner),
        f"{inner}int iteration_changed = 0;",
        _jacobi_projection(rows, pooled=pooled, pad=inner),
        _jacobi_momentum(rows, pad=inner),
        f"{inner}{{",
        f"{inner}    const int i = lane;",
        f"{inner}    if (i < {rows.rows}) {{",
        f"{inner}        s_x[i] = x_new;",
        f"{inner}        s_y[i] = x_new + beta * (x_new - x_old);",
        f"{inner}        if (x_new != x_from) {{",
        change_vote(laws, (change,), inner + "            "),
        f"{inner}        }}",
        f"{inner}    }}",
        f"{inner}}}",
        f"{inner}SYNC();",
        end_sweep(
            policy,
            iteration="sweep",
            block=block,
            iterations=str(int(sweeps)),
            argmax="BLOCK_ARGMAX" if block else "ARGMAX",
        ).rstrip("\n"),
        f"{pad}}}",
    ]
    return "\n".join(line for line in lines if line)


def jacobi_limit_pass(rows: JacobiRows, op, *, pad: str = "    ") -> str:
    """The last word of the joint velocity-limit rows after the final Jacobi sweep, on the first warp, in row order.

    With ``r_i = b'_i + (A x)_i``, the residual the sweeps use, a violated admitted row (``r_i < 0``) is pushed onto
    its limit, ``x_i += -r_i / A_ii`` (the row law's update from a zero impulse,
    :func:`.projection_laws.velocity_limit_update`, added to the row's total). The pass only pushes: it changes
    nothing at the sweeps' fixed point, where no row is violated, and gives the limits the last word of a world
    stopped early, as the Gauss-Seidel sweeps' end pass does. The operator forms ``(A x)_i``
    (:meth:`~.ResponseOperator.push_velocity`, on ``w = Y^T x`` where it keeps one, following each push).
    """
    if not rows.limits:
        raise ValueError("the last word pushes the velocity-limit rows the rows stage")
    warp = Lanes(32)
    body = "            "
    lines = [
        "{",
        "    SYNC();",
        op.push_prologue("    "),
        "    if (lane < 32) {",
        f"        for (int i = 0; i < {rows.rows}; ++i) {{",
        f"            if (s_kind[i] != {JACOBI_UNILATERAL} || s_vlim[i] == 0) continue;",
        op.push_velocity("part", warp, body),
        "            const float r_vlim = s_rhs[i] + part;",
        f"            const float diag_vlim = {op.push_diagonal()};",
        "            const float delta_vlim = diag_vlim > 0.0f",
        "                ? ProjectionLaw::velocity_limit_update(0.0f, r_vlim / diag_vlim).v1 : 0.0f;",
        "            if (delta_vlim != 0.0f) {",
        "                if (lane == 0) s_x[i] += delta_vlim;",
        op.push_apply("delta_vlim", warp, body + "    "),
        "            }",
        "            WARP_SYNC();",
        "        }",
        "    }",
        "    SYNC();",
        "}",
    ]
    return indent("\n".join(line for line in lines if line), pad)

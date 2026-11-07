# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The sweep emitter: FeatherPGS's projected Gauss-Seidel and Jacobi sweeps from one generator.

A sweep is a configuration ``Sweep(mapping, staging, blocks, update, laws, exit, publication)``:

- the **mapping** says which threads own a problem (a world, or a lane group per world) and how many lanes update
  one row together (:class:`.response_operator.Lanes`);
- the **staging** declares the block's shared working set (:class:`.memory.Staging`);
- each **row block** pairs a row store (where a row's impulse, type, parent, friction coefficient, right-hand side
  and diagonal live: :class:`WorldRows`, one accessor for both row classes) with a response operator
  (:mod:`.response_operator`) and the row kinds it visits;
- the **row visit** admits a row of a row loop (:func:`row_visit`): a friction pair at its first tangent (:func:`admit_pair`), the diagonal check and the pair's disk on the patch
  ring's pooled load (:func:`friction_radius`), then the operator's residual;
- the **row law** is one function, :func:`row_update`: step, regularization, the projection of each row kind
  (unilateral, friction pair, velocity limit, bilateral), the change vote (:func:`change_vote`) and the
  apply. Every block of every sweep runs it; a friction pair solved as one unit runs its pair law and tail
  (:func:`pair_update`);
- the **contact unit** is a contact normal and its friction pair visited by the row law on the unit's Gram block
  (:func:`contact_unit`, :class:`UnitZ`): the placement reduces the unit's terms once and applies its changes
  once, the pair's terms moved by the normal's change;
- the **Jacobi law** steps every row of a world at once instead, a lane each (:class:`JacobiRows`,
  :func:`jacobi_sweeps`): the scaled step from the extrapolated impulses, bounded over the rows that can move (an idle
  row joins when its residual turns negative), isotropic in each friction block's scaled coordinates, the radial
  projection of each block onto its disk or ball, Nesterov's momentum with adaptive restart and the relative change
  vote, the operator forming each row's residual from the impulses;
- the **exit** is the exit law (:mod:`.solve_policy`): each change votes against the rounding floor of its update
  (:func:`change_vote`) and the end of the sweep stops a world none of whose updates was loud (:func:`end_sweep`);
- the **publication** writes the impulses and velocities back.

Only the response operator and the lane layout fix a row's floating-point order, so a configuration that keeps a
sweep's operator and layout reproduces its results.
"""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Callable, Sequence
from typing import ClassVar, Protocol

import warp as wp

from ...core.execution import Work, shape
from ...core.memory import Buffer, Staging, arch_budget, built_for
from .device_preludes import BLOCK_THREADS, RING, SHARED_STORAGE, DevicePrelude, VirtualLanes, group_native, lane_group
from .kernel_template import Dispatch, KernelTemplate
from .law_source import law_preludes
from .projection_laws import CONE_LAW, PROJECTION_LAW
from .response_operator import Lanes, LaneVelocity, ResponseOperator, Velocity
from .row_inputs import RowArgument
from .row_kernels import (
    PGS_CONSTRAINT_TYPE_CONTACT,
    PGS_CONSTRAINT_TYPE_FRICTION,
    PGS_CONSTRAINT_TYPE_JOINT_LIMIT,
    PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT,
    PGS_CONSTRAINT_TYPE_SPEED,
    PGS_CONSTRAINT_TYPE_TORSION,
)
from .row_metadata import (
    _DENSE_META_ROW_TYPE_BITS,
    _DENSE_META_ROW_TYPE_MASK,
    _mf_parent_expression,
    load_row_record,
    record_word,
    record_words,
    unpack_mf_header,
)
from .solve_policy import DEFAULT_POLICY, Change, SolvePolicy, bound, sweep_exit, terms_bound, vote

CONTACT = int(PGS_CONSTRAINT_TYPE_CONTACT)
FRICTION = int(PGS_CONSTRAINT_TYPE_FRICTION)
JOINT_LIMIT = int(PGS_CONSTRAINT_TYPE_JOINT_LIMIT)
VELOCITY_LIMIT = int(PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT)
SPEED = int(PGS_CONSTRAINT_TYPE_SPEED)
TORSION = int(PGS_CONSTRAINT_TYPE_TORSION)


def indent(text: str, prefix: str) -> str:
    """Prefix every non-empty line of ``text``."""
    return "\n".join(prefix + line if line.strip() else line for line in text.split("\n"))


# ---------------------------------------------------------------------------------------------------------------------
# Row stores: where a row's state and metadata live
# ---------------------------------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class WorldRows:
    """A world's rows of one class in the row store's line: the dense class's from ``off_dense``, the free bodies'
    (``free``) from ``off_mf``, ``capacity`` slots each, read by the class's staging policy.

    ``staging`` says where a row's metadata come from:

    - ``"shared"``: the right-hand side (the law's bias plus the row's incident velocity ``row_incident``: the row's
      residual at ``du = 0``), diagonal and friction coefficient are staged in shared memory, and the kind and parent
      as one packed word per row (``s_meta_dense``: the kind in the low bits, the parent plus one above);
    - ``"record"``: the rows' records stream (``row_record``, :func:`.row_metadata.pack_row_record`), the class's from
      ``off_rec_dense`` or ``off_rec_mf``. A row loop loads the visited row's record a row ahead into registers
      (``rec_a``, ``rec_b``: :func:`rows_block`, or :meth:`visit`), which every read of row ``i`` takes; any other
      row's read takes its record's word. The free rows always stream their headers (``rec_a``,
      :func:`.row_metadata.pack_row_header`), with their diagonal's inverse; the dense rows' records keep ``A_ii``.

    Impulses are staged (``s_lam_dense``, ``s_lam_mf``) or, with ``stream``, in place (``g_lam_dense``, ``g_lam_mf``).
    A dense row's impulse is owned by its first lane, which every lane reads by broadcast; a staged free row's every
    lane, and every row's with ``every_lane``, stores what it computed and reads its own stores.
    """

    capacity: int
    staging: str = "shared"
    free: bool = False
    stream: bool = False
    rhs: str = "rhs_bias"
    lanes: Lanes = dataclasses.field(default_factory=Lanes)
    every_lane: bool = False

    STAGINGS: ClassVar[tuple[str, ...]] = ("shared", "record")
    STAGED: ClassVar[dict[str, str]] = {
        "s_rhs_dense": "rhs",
        "s_diag_dense": "world_diag",
        "s_mu_dense": "world_row_mu",
    }
    VISITED: ClassVar[str] = "i"
    """The row a row loop visits, whose record the loop holds in registers."""

    def __post_init__(self):
        if self.staging not in self.STAGINGS:
            raise ValueError(f"staging must be one of {self.STAGINGS}, got {self.staging!r}")
        if self.free and self.staging != "record":
            raise ValueError("the free bodies' rows stream their records")
        if self.stream and self.every_lane:
            raise ValueError("a class whose every lane reads its own stores stages its impulses")

    @property
    def record(self) -> bool:
        return self.staging == "record"

    @property
    def own_stores(self) -> bool:
        """Every lane stores the impulse it computed and reads its own stores (a staged free row, ``every_lane``)."""
        return self.every_lane or (self.free and not self.stream)

    @property
    def owned(self) -> bool:
        """The row's first lane owns its stored impulse, which other lanes read directly: its stores wait for them.
        A free row's every read takes its own lane's store or the first lane's by broadcast."""
        return not (self.free or self.every_lane)

    @property
    def base(self) -> str:
        """The class's first row in the world's line."""
        return "off_mf" if self.free else "off_dense"

    @property
    def records(self) -> str:
        """The class's first record in the line's records."""
        return "off_rec_mf" if self.free else "off_rec_dense"

    @property
    def impulses(self) -> str:
        """The staged impulses (or, with ``stream``, the world's in place)."""
        return f"{'g' if self.stream else 's'}_lam_{'mf' if self.free else 'dense'}"

    @property
    def words(self) -> int:
        """Words of a row's record (:func:`.row_metadata.record_words`): a free row's header, a dense row's two."""
        return record_words(self.free)

    @property
    def registers(self) -> tuple[str, ...]:
        """The visited row's record registers: ``rec_a`` (the header), and ``rec_b`` past it."""
        return ("rec_a",) if self.free else ("rec_a", "rec_b")

    def word(self, row: str, word: int) -> str:
        """Word ``word`` of row ``row``'s record: the visited row's from its registers."""
        if row == self.VISITED:
            if word >= self.words:
                raise ValueError(f"a record of {self.words} words has no word {word}")
            return f"rec_{'ab'[word // 4]}.{'xyzw'[word % 4]}"
        return record_word(row, word, self.records, self.words)

    def load_record(self, names: tuple[str, ...], row: str, indent: str) -> str:
        """Load row ``row``'s record into the ``int4`` locals ``names`` (one a header's, two a dense record's)."""
        return load_row_record(names[0], names[1] if len(names) > 1 else None, row, self.records, indent, self.words)

    def visit(self, row: str = "i") -> str:
        """Declare the visited row's record registers for a loop that does not load them a row ahead."""
        if not self.record:
            return ""
        return f"int4 {', '.join(self.registers)};\n" + self.load_record(self.registers, row, "")

    def _source(self, name: str) -> str:
        source = self.STAGED[name]
        return self.rhs if source == "rhs" else source

    def _stored(self, name: str, row: str) -> str:
        """Row ``row``'s value of the staged array ``name`` where it is stored."""
        value = f"{self._source(name)}.data[off_dense + ({row})]"
        return f"({value} + row_incident.data[off_dense + ({row})])" if name == "s_rhs_dense" else value

    def staged(self, name: str, row: str) -> str:
        """Row ``row``'s value of the staged array ``name``."""
        return f"{name}[{row}]"

    def rhs_of(self, row: str) -> str:
        if self.record:
            return f"__int_as_float({self.word(row, 2)})"
        return self.staged("s_rhs_dense", row)

    def diag(self, row: str) -> str:
        if self.record:
            return f"__int_as_float({self.word(row, 1)})"
        return self.staged("s_diag_dense", row)

    def inverse(self, row: str) -> str:
        """A free row's stored inverse diagonal."""
        return f"__int_as_float({self.word(row, 1)})"

    def diagonal(self, own: str) -> Diagonal:
        """The rows' diagonals as the class keeps them (the visited row ``i``'s read into ``own``): a free row's
        stored inverse, which its step multiplies by, or the dense rows' value, which a step divides by."""
        if self.free:
            return Diagonal(inverse=lambda row: own if row == "i" else self.inverse(row))
        return Diagonal(lambda row: own if row == "i" else self.diag(row))

    def mu(self, row: str) -> str:
        if self.free:
            return f"world_row_mu.data[off_mf + {row}]"
        if self.record:
            return f"__int_as_float({self.word(row, 5)})"
        return self.staged("s_mu_dense", row)

    def weight(self, row: str) -> str:
        return f"world_row_w.data[{self.base} + {row}]"

    def eps(self, row: str) -> str:
        return f"world_row_eps.data[{self.base} + {row}]"

    def kind(self, row: str, *, scan: bool = False) -> str:
        if self.record:
            return f"({self.word(row, 3)} & 0xFFFF)"
        return f"s_meta_dense[{row}] & {_DENSE_META_ROW_TYPE_MASK}"

    def parent(self, row: str, *, enclosed: bool = False) -> str:
        decoded = (
            self.parent_of(self.word(row, 3))
            if self.record
            else f"(s_meta_dense[{row}] >> {_DENSE_META_ROW_TYPE_BITS}) - 1"
        )
        return f"({decoded})" if enclosed else decoded

    def parent_of(self, packed: str) -> str:
        """A row's parent from its record's packed ``kind | parent << 16`` word."""
        return _mf_parent_expression(packed, self.capacity)

    def scan(self) -> tuple[Callable[[str], str], Callable[[str], str]]:
        """A row's kind and parent as a scan over the rows reads them (:func:`friction_metric_stage`)."""
        return (lambda row: f"({self.kind(row, scan=True)})"), self.parent

    def spin_of(self, parent: str, end: str) -> str:
        """The spin row of the contact whose normal is row ``parent``, of rows ``[0, end)`` (:func:`spin_row`)."""
        return spin_row(parent, end, lambda row: f"({self.kind(row)})")

    def load(self, row: str) -> str:
        """Row ``row``'s impulse on every lane (read by the first lane where it owns it)."""
        if self.own_stores:
            return self.peek(row)
        return self.lanes.broadcast(f"{self.impulses}[{row}]")

    def peek(self, row: str) -> str:
        """Row ``row``'s impulse read directly (every lane reads the staged value)."""
        return f"{self.impulses}[{row}]"

    def store(self, row: str, value: str) -> str:
        if self.own_stores:
            return f"{self.impulses}[{row}] = {value};"
        return self.lanes.first(f"{self.impulses}[{row}] = {value};")

    def ring_load(self, parent: str, load: str | None = None) -> str:
        """The pooled normal load of the patch ring of normal row ``parent``, from ``load`` (its impulse)."""
        if self.free:
            member = self.parent("patch_row")
            impulse = self.load("patch_row")
            return (
                f"RING_LOAD({load or self.load(parent)}, {parent}, {self.capacity},\n"
                f"    [&](int patch_row) {{ return {member}; }},\n"
                f"    [&](int patch_row) {{ return {impulse}; }})"
            )
        return (
            f"RING_LOAD({load or self.peek(parent)}, {parent}, {self.capacity},\n"
            f"    [&](int patch_row) {{ return {self.parent('patch_row', enclosed=True)}; }},\n"
            f"    [&](int patch_row) {{ return {self.peek('patch_row')}; }})"
        )

    # -- staging ---------------------------------------------------------------------------------------------------

    def staged_names(self) -> tuple[list[str], list[str]]:
        return (["s_rhs_dense", "s_diag_dense"], ["s_mu_dense"]) if self.staging == "shared" else ([], [])

    def buffers(self) -> tuple[Buffer, ...]:
        impulses = () if self.stream else (Buffer(self.impulses, "float", self.capacity),)
        if self.free:
            return impulses
        head, tail = self.staged_names()
        packed = [Buffer("s_meta_dense", "int", self.capacity)] if self.staging == "shared" else []
        return (
            *impulses,
            *(Buffer(name, "float", self.capacity) for name in head),
            *packed,
            *(Buffer(name, "float", self.capacity) for name in tail),
        )

    def declaration(self) -> str:
        """The streamed impulses' view of the world's line."""
        if self.stream:
            return f"    volatile float* {self.impulses} = (volatile float*)(&world_impulses.data[{self.base}]);"
        return ""

    def loads(self, row: str = "i") -> str:
        """Stage row ``row``'s impulse and metadata."""
        lines = [] if self.stream else [f"{self.impulses}[{row}] = world_impulses.data[{self.base} + {row}];"]
        if self.record:
            return "\n".join(lines)
        head, tail = self.staged_names()
        lines += [f"{name}[{row}] = {self._stored(name, row)};" for name in head]
        lines += [
            f"int row_type = world_row_type.data[off_dense + {row}];",
            f"int row_parent = world_row_parent.data[off_dense + {row}];",
            f"s_meta_dense[{row}] = (row_type & {_DENSE_META_ROW_TYPE_MASK}) | "
            f"((row_parent + 1) << {_DENSE_META_ROW_TYPE_BITS});",
        ]
        lines += [f"{name}[{row}] = {self._stored(name, row)};" for name in tail]
        return "\n".join(lines)

    def stores(self, row: str = "i") -> str:
        """Publish row ``row``'s staged impulse."""
        return "" if self.stream else f"world_impulses.data[{self.base} + {row}] = {self.impulses}[{row}];"


def spin_row(parent: str, end: str, kind: Callable[[str], str], *, spin: str = str(TORSION)) -> str:
    """The rows of a contact, read from the data: the spin row of the contact whose normal is row ``parent``, or -1.

    A friction contact's rows are its normal, its pair and, where its region has a footprint or contact torsion gives
    every contact one, its spin row (:func:`.friction_patches.patch_spin_rows`), in that order. A spin row only ever
    follows its own pair, so the contact has one exactly when the row after the pair, ``parent + 3``, is one: ``kind``
    reads a row's kind from the sweep's row store (``spin`` the kind of a spin row there), and ``end`` bounds the rows it
    holds. Every sweep whose laws allow spin rows reads a contact's rows through this test, and a contact without a spin
    row solves its pair as the triple of a zero spin coefficient, which is the pair's law bit for bit
    (:func:`.projection_laws.friction_ball`).
    """
    row = f"({parent}) + 3"
    return f"({row} < ({end}) && {kind(row)} == {spin} ? {row} : -1)"


class RowStore(Protocol):
    """Where a world's rows live, as the row law and a sweep's phases read them: every accessor is native text at a
    row index. Two stores answer it: :class:`WorldRows` (a class's rows in the row store's line, staged or streamed by
    the class's policy: the matrix-free and sparse-diagonal sweeps) and :class:`StagedRows` (a world's rows staged
    whole: the sparse and paired factor sweeps)."""

    lanes: Lanes
    owned: bool
    """The row's first lane owns its stored impulse, which other lanes read: its stores wait for their reads
    (:attr:`RowContext.owned`). Otherwise every lane reads its own stores, or the first lane's by broadcast."""

    def load(self, row: str) -> str:
        """Row ``row``'s impulse on every lane."""

    def peek(self, row: str) -> str:
        """Row ``row``'s impulse as stored."""

    def store(self, row: str, value: str) -> str:
        """Row ``row``'s impulse store, by the lanes that own it."""

    def rhs_of(self, row: str) -> str: ...

    def diag(self, row: str) -> str: ...

    def diagonal(self, own: str) -> Diagonal:
        """The rows' diagonals as the store keeps them (:class:`Diagonal`), the visited row ``i``'s read into ``own``:
        a stored reciprocal is multiplied by, a value divided by."""

    def mu(self, row: str) -> str: ...

    def weight(self, row: str) -> str: ...

    def eps(self, row: str) -> str: ...

    def kind(self, row: str, *, scan: bool = False) -> str: ...

    def parent(self, row: str, *, enclosed: bool = False) -> str: ...

    def ring_load(self, parent: str, load: str | None = None) -> str:
        """The pooled normal load of the patch ring of normal row ``parent``, from ``load`` (its impulse)."""

    def spin_of(self, parent: str, end: str) -> str:
        """The spin row of the contact whose normal is row ``parent``, of rows ``[0, end)`` (:func:`spin_row`)."""

    def scan(self) -> tuple[Callable[[str], str], Callable[[str], str]]:
        """A row's kind and parent as a scan over the rows reads them (:func:`friction_metric_stage`)."""

    def visit(self, row: str = "i") -> str:
        """What a loop over the rows declares before it reads row ``row`` (a streamed record's registers)."""


@dataclasses.dataclass(frozen=True)
class StagedRows:
    """A world's rows staged whole (the sparse and paired factor sweeps' store).

    Impulses, right-hand sides (``rhs_bias + row_incident``), diagonals and kinds are staged per world (``s_impulse``,
    ``s_rhs``, ``s_diag``, ``s_type``); every lane reads them and the first lane stores the impulses. With
    ``reciprocal`` the diagonals' reciprocals are staged too (``s_inverse``): a row's diagonal is fixed for the solve,
    so its reciprocal is formed once, at staging, and a row's step multiplies by it (the exit law reads the diagonal
    itself); without, a step divides. Parents, friction coefficients, weights and compliances stream from the row
    store's line from ``line``, the kernel's local for the world's first row; ``count`` is its local for the world's
    row count, which bounds a patch ring's walk.

    With ``triples`` (contact torsion, a sweep that solves friction triples in a phase of their own) a triple's pair
    (a contact with a spin row, :func:`spin_row`) is staged as the spin row's kind: the row loop and the active scan
    skip the triple's three rows, and the triple phase finds its normal by the staged kinds.
    """

    capacity: int
    lanes: Lanes
    line: str = "offset"
    count: str = "count"
    reciprocal: bool = True
    triples: bool = False

    owned: ClassVar[bool] = True
    """The first lane stores the staged impulses, which every lane reads."""

    def load(self, row: str) -> str:
        return f"s_impulse[{row}]"

    def peek(self, row: str) -> str:
        return f"s_impulse[{row}]"

    def store(self, row: str, value: str) -> str:
        return self.lanes.first(f"s_impulse[{row}] = {value};")

    def rhs_of(self, row: str) -> str:
        return f"s_rhs[{row}]"

    def diag(self, row: str) -> str:
        return f"s_diag[{row}]"

    def inverse(self, row: str) -> str:
        return f"s_inverse[{row}]"

    def diagonal(self, own: str) -> Diagonal:
        return Diagonal(lambda row: own if row == "i" else self.diag(row), self.inverse if self.reciprocal else None)

    def kind(self, row: str, *, scan: bool = False) -> str:
        return f"s_type[{row}]"

    def parent(self, row: str, *, enclosed: bool = False) -> str:
        return f"world_row_parent.data[{self.line} + {row}]"

    def mu(self, row: str) -> str:
        return f"world_row_mu.data[{self.line} + {row}]"

    def weight(self, row: str) -> str:
        return f"world_row_w.data[{self.line} + {row}]"

    def eps(self, row: str) -> str:
        return f"world_row_eps.data[{self.line} + {row}]"

    def ring_load(self, parent: str, load: str | None = None) -> str:
        return f"RING_LOAD({load or self.load(parent)}, {parent}, {self.count}, ring_parent, ring_impulse)"

    def visit(self, row: str = "i") -> str:
        return ""

    def spin_of(self, parent: str, end: str) -> str:
        return spin_row(parent, end, lambda row: f"static_cast<int>({self.kind(row)})")

    def scan(self) -> tuple[Callable[[str], str], Callable[[str], str]]:
        return (lambda row: f"static_cast<int>({self.kind(row)})"), self.parent

    # -- staging ---------------------------------------------------------------------------------------------------

    def buffers(self) -> tuple[Buffer, ...]:
        rows = self.capacity
        inverse = (Buffer("s_inverse", "float", rows),) if self.reciprocal else ()
        return (
            Buffer("s_impulse", "float", rows),
            Buffer("s_rhs", "float", rows),
            Buffer("s_diag", "float", rows),
            *inverse,
            Buffer("s_type", "unsigned char", rows),
        )

    def rings(self, pad: str) -> str:
        """The patch rings' accessors :meth:`ring_load` walks."""
        return (
            f"{pad}// A contact normal's patch ring links the region's next normal (-1 ends a point contact).\n"
            f"{pad}const auto ring_parent = [&](int r) {{ return {self.parent('r')}; }};\n"
            f"{pad}const auto ring_impulse = [&](int r) {{ return {self.load('r')}; }};"
        )

    def loads(self, pad: str) -> str:
        """Stage the world's rows, a lane a row (with ``triples``, ``world_triples``: whether the world holds one)."""
        lane, width, line, count = self.lanes.lane, self.lanes.width, self.line, self.count
        kind = f"static_cast<unsigned char>(world_row_type.data[{line} + i])"
        triple = ""
        if self.triples:
            kind = "static_cast<unsigned char>(kind)"
            spin = spin_row("parent", count, lambda row: f"world_row_type.data[{line} + {row}]")
            triple = (
                f"{pad}    int kind = world_row_type.data[{line} + i];\n"
                f"{pad}    if (kind == {FRICTION}) {{\n"
                f"{pad}        const int parent = world_row_parent.data[{line} + i];\n"
                f"{pad}        if ({spin} >= 0) kind = {TORSION};\n"
                f"{pad}    }}\n"
                f"{pad}    if (kind == {TORSION}) staged_triples = true;\n"
            )
        diagonal = f"{pad}    s_diag[i] = world_diag.data[{line} + i];\n"
        if self.reciprocal:
            diagonal = (
                f"{pad}    const float diagonal = world_diag.data[{line} + i];\n"
                f"{pad}    s_diag[i] = diagonal;\n"
                f"{pad}    s_inverse[i] = 1.0f / diagonal;\n"
            )
        # Whether the world holds a friction triple (a spin row): the triple phase runs only where one does.
        present = f"\n{pad}const bool world_triples = ANY(staged_triples);\n" if self.triples else ""
        return (
            (f"{pad}bool staged_triples = false;\n" if self.triples else "")
            + f"{pad}for (int i = {lane}; i < {count}; i += {width}) {{\n"
            f"{pad}    s_impulse[i] = world_impulses.data[{line} + i];\n"
            f"{pad}    s_rhs[i] = rhs_bias.data[{line} + i] + row_incident.data[{line} + i];\n"
            f"{diagonal}{triple}{pad}    s_type[i] = {kind};\n"
            f"{pad}}}{present}"
        )

    def stores(self, pad: str) -> str:
        """Publish the world's staged impulses."""
        lane, width = self.lanes.lane, self.lanes.width
        return f"{pad}for (int i = {lane}; i < {self.count}; i += {width}) world_impulses.data[{self.line} + i] = s_impulse[i];"


# ---------------------------------------------------------------------------------------------------------------------
# The row law
# ---------------------------------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class RowLaws:
    """The physics a sweep's rows obey, beyond each row kind's projection.

    Args:
        regularize: Contact rows take their regularization weight (the laws are weighted): a constant of the step's
            laws, compiled into every sweep.
        material: Material contacts exist: a contact row of compliance ``eps > 0`` takes the weight ``d / (d + eps)``
            from the diagonal the placement forms (:func:`row_weight`), its stored one otherwise.
        torsion: Spin rows may follow friction pairs (:func:`spin_row`): the row loops skip the spin rows, and a
            pair's visit solves the friction triple, the pair and its contact's spin row (if any) on their elliptic
            cone (:func:`friction_pair`).
        policy: The solve policy: the readout beside the exit law.
    """

    regularize: bool = True
    material: bool = False
    torsion: bool = False
    policy: SolvePolicy = DEFAULT_POLICY


def row_weight(weight: str, eps: str, diagonal: str) -> str:
    """A contact row's regularization weight: ``d / (d + eps)`` for a material row of compliance ``eps > 0`` and
    diagonal ``d``, else its stored ``weight`` (the relative option's ``1 / (1 + g)``, or 1).

    The regularized step (:func:`.projection_laws.regularized_delta`) with ``d / (d + eps)`` is the material row's
    exact Gauss-Seidel step, ``-(r + eps lambda) / (d + eps)``, whichever diagonal the placement forms. Emitted only
    where the laws have material contacts, so no other sweep's text changes.
    """
    return f"({eps} > 0.0f ? ({diagonal}) / (({diagonal}) + {eps}) : {weight})"


def weight_arguments(weighted: bool) -> tuple[RowArgument, ...]:
    """The operands a sweep whose ABI holds no row weights takes once its laws are weighted: the rows' weights."""
    return (RowArgument("world_row_w", wp.array2d[float]),) if weighted else ()


def material_arguments(material: bool) -> tuple[RowArgument, ...]:
    """The operands of a sweep with material contacts: the rows' compliance (the row store's, both classes'), zero but
    on a material contact's normal."""
    if not material:
        return ()
    return (RowArgument("world_row_eps", wp.array2d[float]),)


def change_vote(laws: RowLaws, changes: tuple[Change, ...], pad: str) -> str:
    """The exit law's vote on one update's ``changes`` (:func:`.solve_policy.vote`): a loud change keeps the world
    sweeping."""
    return vote(laws.policy, changes, flag="iteration_changed", pad=pad)


@dataclasses.dataclass(frozen=True)
class Diagonal:
    """A row's diagonal ``A_ii`` as its row store keeps it, which the row law reads: the value (``diagonal``) and,
    where the store keeps or the sweep stages it, its reciprocal (``inverse``), each a callable from a row index to the
    native expression.

    The Gauss-Seidel step multiplies by the reciprocal when there is one and divides by the value otherwise; the exit
    law reads the value when there is one (velocity units) and the reciprocal otherwise (impulse units).
    """

    diagonal: Callable[[str], str] | None = None
    inverse: Callable[[str], str] | None = None

    def value(self, row: str) -> str:
        """``A_ii`` of row ``row`` (zero for a zero reciprocal)."""
        if self.diagonal is not None:
            return self.diagonal(row)
        inverse = self.inverse(row)
        return f"{inverse} > 0.0f ? 1.0f / {inverse} : 0.0f"

    def bound(self, rhs: str, terms: str, impulse: str, row: str) -> str:
        """The exit law's bound of row ``row`` at ``impulse`` (:func:`.solve_policy.bound`), in the form the store
        gives without a division: velocity units from a value, impulse units from a reciprocal."""
        if self.diagonal is not None:
            return bound(rhs, terms, impulse, diagonal=self.diagonal(row))
        return bound(rhs, terms, impulse, step=self.inverse(row))

    def change(self, delta: str, bound: str, kind: str, row: str) -> Change:
        """Row ``row``'s change ``delta`` against its :meth:`bound`, as the exit law votes it."""
        return Change(delta, bound, kind, self.value(row), scaled=self.diagonal is not None)

    def step(self, residual: str, row: str) -> str:
        """The Gauss-Seidel step ``-r / A_ii``."""
        if self.inverse is not None:
            return f"-{residual} * {self.inverse(row)}"
        return f"-{residual} / {self.diagonal(row)}"

    def scale(self, residual: str, row: str) -> str:
        """``r / A_ii`` (the step's magnitude, as the velocity-limit law takes it)."""
        if self.inverse is not None:
            return f"{residual} * {self.inverse(row)}"
        return f"{residual} / {self.diagonal(row)}"


@dataclasses.dataclass(frozen=True, kw_only=True)
class RowContext:
    """What :func:`row_update` reads of a row: its kind, impulse, metadata, diagonal and response operator.

    The expressions are native code at the row index ``row``; ``kind`` is a
    local the block has already declared, or a known kind. Only friction rows read a row's friction coefficient,
    parent and patch ring (``mu``, ``parent``, ``ring_load``), and only weighted laws its weight.
    """

    row: str
    kind: str
    operator: ResponseOperator
    diagonal: Diagonal
    """The rows' diagonals as the row store keeps them."""
    load: Callable[[str], str]
    store: Callable[[str, str], str]
    rhs: Callable[[str], str]
    lanes: Lanes
    mu: Callable[[str], str] | None = None
    weight: Callable[[str], str] | None = None
    parent: str | None = None
    ring_load: Callable[..., str] | None = None
    """The pooled normal load of the patch ring of normal row ``parent``: ``ring_load(parent)`` from the row's stored
    impulse, ``ring_load(parent, load)`` from ``load`` (a contact unit's normal in a register, :func:`contact_unit`)."""
    eps: Callable[[str], str] | None = None
    """A row's compliance, where the laws have material contacts (:attr:`RowLaws.material`)."""
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
    triple_law: str = "ConeLaw::friction_triple("
    """The friction triple's law, up to its first argument: :func:`.projection_laws.friction_triple`, or a law of the
    same arguments after a leading ``int`` of its own (a cache of the triple's metric, :func:`cached_triple_law`)."""
    owned: bool = True
    """The row's first lane owns the stored impulses: the lanes synchronize before the stores (every lane has read
    the impulses the first lane overwrites; :attr:`first_store`); the applies after them read no impulse, and the
    row's closing barrier (:meth:`sync`) orders the stores before the next row's reads. Otherwise every lane stores
    what it computed, as every other lane does, and reads only its own stores."""
    first_store: str | None = None
    """For rows that read their impulses from registers loaded before them (a contact unit's, :func:`contact_unit`),
    the condition under which a row's stores are the first since those loads: only these wait for the lanes. ``None``:
    every row's stores wait."""
    synced: bool = True
    """The lanes synchronize after a row's stores and apply (:meth:`sync`). A contact unit's rows defer their applies
    and read no store of the unit, so its placement synchronizes once, after the unit's apply."""
    spin_row: Callable[[str], str] | None = None
    """The spin row of the contact whose normal is row ``parent`` (-1 for none), as the row store reads it
    (:func:`spin_row`), where the laws allow spin rows."""
    pair_residual: bool = False
    """A friction pair reduces its first tangent's residual again with the pair's terms
    (:func:`_pair_terms` with ``own``) and solves on it in place of the row's: the free-body loop's
    friction step."""

    @classmethod
    def of(
        cls, rows: RowStore, op, *, end: str, own: str = "denom", metric: str | None = None, **choices
    ) -> RowContext:
        """The context of row ``i`` of a row loop over a store's rows ``[0, end)``: every row accessor is the store's
        (:class:`RowStore`), its ownership included, the visited row's diagonal read into the local ``own``, its kind
        into ``row_type`` and a friction row's parent into ``parent``; with ``metric`` a friction triple's law reads
        its metric cached for the launch (:func:`friction_metric_law`). ``choices`` sets the context's other fields."""
        return cls(
            row="i",
            kind="row_type",
            operator=op,
            diagonal=rows.diagonal(own),
            load=rows.load,
            store=rows.store,
            rhs=rows.rhs_of,
            mu=rows.mu,
            weight=rows.weight,
            eps=rows.eps,
            parent="parent",
            ring_load=rows.ring_load,
            lanes=rows.lanes,
            owned=rows.owned,
            spin_row=lambda contact: rows.spin_of(contact, end),
            **({"triple_law": f"{metric}(parent, "} if metric is not None else {}),
            **choices,
        )

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
    op, diag, row, kind = ctx.operator, ctx.diagonal, ctx.row, ctx.kind
    lines = [
        f"float old_impulse = {ctx.load(row)};",
        f"float residual = {residual} + {ctx.rhs(row)};",
        f"float row_bound = {_row_bound(ctx, op.terms(residual, ctx.rhs(row), whole=False))};",
    ]
    step = diag.step("residual", row)
    if laws.regularize:
        weight = ctx.weight(row)
        if laws.material:
            weight = row_weight(weight, ctx.eps(row), diag.value(row))
        lines += [
            f"float delta = {step};",
            f"if ({kind} == {CONTACT})",
            f"    delta = ProjectionLaw::regularized_delta(delta, {weight}, old_impulse);",
            "float new_impulse = ProjectionLaw::relaxed_step(old_impulse, delta, omega);",
        ]
    else:
        lines.append(f"float new_impulse = ProjectionLaw::relaxed_step(old_impulse, {step}, omega);")
    lines.append("float delta_impulse = 0.0f;")
    if ctx.friction:
        lines += ["int sib = -1;", "float sib_new = 0.0f, sib_delta = 0.0f, sib_bound = 0.0f;"]
        if laws.torsion:
            # The contact's spin row (``spin``, -1 for none) is the visit's (:func:`admit_pair`).
            lines += ["float spin_new = 0.0f, spin_delta = 0.0f, spin_bound = 0.0f;"]
    branches = []
    if ctx.inline_velocity_limit:
        branches.append(
            (
                f"{kind} == {VELOCITY_LIMIT}",
                f"const auto vlim = ProjectionLaw::velocity_limit_update(old_impulse, {diag.scale('residual', row)});\n"
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
        pair = friction_pair(ctx, laws, own=ctx.pair_residual)
        if _sibling_in_pair(ctx):
            # Impulses that need no barrier: the pair stores, votes and applies its second tangent (and spin row) in
            # its branch, so the rows without a pair carry no sibling test.
            siblings = [("sib", "sib_new", "sib_delta", _sibling_change(ctx))]
            if laws.torsion:
                siblings.append(("spin", "spin_new", "spin_delta", _spin_change(ctx)))
            # A contact without a spin row stores nothing for it (spin_row).
            guard = {"spin": "if (spin >= 0) "}
            pair += "\n" + "\n".join(
                line
                for sibling, new, delta, change in siblings
                for line in (
                    guard.get(sibling, "") + ctx.store(sibling, new),
                    (guard.get(sibling, "") + ctx.record(sibling, delta)) if ctx.record is not None else "",
                    f"if ({delta} != 0.0f) {{",
                    change_vote(laws, (change,), "    "),
                    indent(op.axpy(((sibling, delta),), "", row=sibling).lstrip("\n"), "    "),
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
    return ctx.diagonal.bound(ctx.rhs(ctx.row), terms, "old_impulse", ctx.row)


def _sibling_change(ctx: RowContext) -> Change:
    """The friction pair's second-tangent change as the exit law reads it (``sib_bound``: :func:`friction_pair`)."""
    return ctx.diagonal.change("sib_delta", "sib_bound", str(FRICTION), "sib")


def _spin_change(ctx: RowContext) -> Change:
    """The spin row's change as the exit law reads it (``spin_bound``: :func:`friction_pair` under torsion)."""
    return ctx.diagonal.change("spin_delta", "spin_bound", str(TORSION), "spin")


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
            ("    if (spin >= 0) " + ctx.store("spin", "spin_new")) if laws.torsion else "",
            ("    if (spin >= 0) " + ctx.record("spin", "spin_delta"))
            if laws.torsion and ctx.record is not None
            else "",
            "}",
        ]
    if ctx.friction and laws.torsion and not _sibling_in_pair(ctx):
        # The spin row's change, ahead of the pair's (its own apply: the triple's rows apply one at a time).
        lines += [
            "if (spin_delta != 0.0f) {",
            change_vote(laws, (_spin_change(ctx),), "    "),
            indent(op.axpy((("spin", "spin_delta"),), "", row="spin").lstrip("\n"), "    "),
            "}",
        ]
    row_change = ctx.diagonal.change("delta_impulse", "row_bound", kind, row)
    sib_change = _sibling_change(ctx)
    if ctx.friction and op.fused_pair_apply:
        # Both changes vote together, the sibling's only where the row has a pair (sib < 0: no sibling row to read).
        sib_change = dataclasses.replace(sib_change, exists="sib >= 0")
        lines += [
            "if (delta_impulse != 0.0f || sib_delta != 0.0f) {",
            change_vote(laws, (row_change, sib_change), "    "),
            indent(op.axpy((("sib", "sib_delta"), ("row", "delta_impulse")), "").lstrip("\n"), "    "),
            "}",
        ]
    else:
        if ctx.friction and not _sibling_in_pair(ctx):
            lines += [
                "if (sib_delta != 0.0f) {",
                change_vote(laws, (sib_change,), "    "),
                indent(op.axpy((("sib", "sib_delta"),), "", row="sib").lstrip("\n"), "    "),
                "}",
            ]
        lines += [
            "if (delta_impulse != 0.0f) {",
            change_vote(laws, (row_change,), "    "),
            indent(op.axpy((("row", "delta_impulse"),), "").lstrip("\n"), "    "),
            "}",
        ]
    lines.append(ctx.sync() if ctx.synced else "")
    return lines


def pair_update(ctx: RowContext, laws: RowLaws, *, pad: str) -> str:
    """The Gauss-Seidel update of a friction pair visited as one unit, at its first tangent ``ctx.row``.

    A unit schedule solves a contact's tangent pair right after its normal
    row ``ctx.parent``, on the disk of ``radius`` (declared by the block): the
    operator reduces the first tangent's residual with the pair's terms
    (:func:`_pair_terms`), the pair is projected as in
    :func:`row_update`, and the same tail stores, votes and applies.
    """
    lines = [
        f"const float old_impulse = {ctx.load(ctx.row)};",
        "float new_impulse = old_impulse;",
        "float delta_impulse = 0.0f;",
        "int sib = -1;",
        "float sib_new = 0.0f, sib_delta = 0.0f, row_bound = 0.0f, sib_bound = 0.0f;",
        *(
            (
                f"const int spin = {ctx.spin_row(ctx.parent)};",
                "float spin_new = 0.0f, spin_delta = 0.0f, spin_bound = 0.0f;",
            )
            if laws.torsion
            else ()
        ),
        friction_pair(ctx, laws, own=True),
        "delta_impulse = new_impulse - old_impulse;",
        *_row_tail(ctx, laws),
    ]
    return indent("\n".join(line for line in lines if line != ""), pad)


def _sibling_bound(ctx: RowContext) -> str:
    """The friction sibling's bound from its residual (``sibling_residual``), at its old impulse."""
    op, rhs = ctx.operator, ctx.rhs("sib")
    return ctx.diagonal.bound(rhs, op.terms("sibling_residual", rhs, whole=True), "sibling_impulse", "sib")


def _spin_bound(ctx: RowContext) -> str:
    """The spin row's bound from its residual (``spin_residual``), at its old impulse."""
    op, rhs = ctx.operator, ctx.rhs("spin")
    return ctx.diagonal.bound(rhs, op.terms("spin_residual", rhs, whole=True), "spin_impulse", "spin")


def _pair_terms(ctx: RowContext, *, own: bool) -> str:
    """Declare a friction pair's ``sibling_residual`` (``Z_s . u + rhs_s``) and ``cross`` (``Z_t . Z_s``), and with
    ``own`` the first tangent's ``residual`` (``Z_t . u + rhs_t``, a pair solved as one unit): one ladder of the
    operator's, or the sums the visit's residual reduced with its own (:attr:`.ResponseOperator.pair_with_visit`)."""
    op, rhs = ctx.operator, ctx.rhs
    sums = ((("residual", "row", None),) if own else ()) + (("sibling_residual", "sib", None), ("cross", "row", "sib"))
    added = {**({"residual": rhs(ctx.row)} if own else {}), "sibling_residual": rhs("sib")}
    reduced = {"sibling_residual": "sibling_jv", "cross": "pair_coupling"} if op.pair_with_visit else None
    return op.dot(sums, {"row": ctx.row, "sib": "sib"}, "    ", rhs=added, reduced=reduced)


def _spin_terms(ctx: RowContext) -> str:
    """Declare a friction triple's spin terms (contact torsion), after :func:`_pair_terms`: ``spin_residual``
    (``Z_p . u + rhs_p``) and its couplings with the tangents, ``spin_cross`` (``Z_t . Z_p``) and
    ``spin_cross_sibling`` (``Z_s . Z_p``), in one ladder on the support and entries the pair's kept."""
    sums = (("spin_residual", "spin", None), ("spin_cross", "row", "spin"), ("spin_cross_sibling", "sib", "spin"))
    rows = {"row": ctx.row, "sib": "sib", "spin": "spin"}
    return ctx.operator.dot(sums, rows, "    ", kept=("row", "sib"), rhs={"spin_residual": ctx.rhs("spin")})


def friction_pair(ctx: RowContext, laws: RowLaws, *, own: bool = False) -> str:
    """The friction pair at its first tangent: both tangents on the disk of ``radius`` (``sib_new``, ``sib_delta``).

    With ``own`` the first tangent's residual is reduced with the pair's terms (a pair solved as a unit). Where spin
    rows may appear (``laws.torsion``) a contact with a spin row (``spin``, after the pair: :func:`spin_row`) solves
    its pair and spin row as the friction triple on its elliptic cone of the pooled normal ``load``
    (:func:`.projection_laws.friction_triple`, the cone's one law every sweep calls): the spin row's coefficient is its
    ``mu`` field, its terms the operator's (:func:`_spin_terms`). A contact without one keeps the
    pair's own law and code, which the triple of a zero spin coefficient equals.
    """
    op = ctx.operator
    head = [
        f"sib = {ctx.parent} + 2;",
        f"const float sibling_impulse = {ctx.load('sib')};",
    ]
    pair = [
        "wp::vec2 pair(0.0f);",
        "// Zero load gives a zero disk; avoid unused tangent reductions.",
        "if (radius > 0.0f) {",
        _pair_terms(ctx, own=own),
        f"    row_bound = {_row_bound(ctx, op.terms('residual', ctx.rhs(ctx.row), whole=True))};" if own else "",
        f"    sib_bound = {_sibling_bound(ctx)};",
        f"    pair = {ctx.pair_law}{ctx.diagonal.value(ctx.row)}, cross, {ctx.diagonal.value('sib')},",
        "        wp::vec2(residual, sibling_residual), wp::vec2(old_impulse, sibling_impulse), radius, omega);",
        "}",
        "pair = ProjectionLaw::disk(pair, radius);",
        "new_impulse = pair[0];",
        "sib_new = pair[1];",
        "sib_delta = sib_new - sibling_impulse;",
    ]
    pair = "\n".join(line for line in pair if line != "")
    if laws.torsion:
        triple = _friction_triple(ctx, own=own)
        pair = f"if (spin < 0) {{\n{indent(pair, '    ')}\n}} else {{\n{indent(triple, '    ')}\n}}"
    return "\n".join(line for line in (*head, pair) if line != "")


def _friction_triple(ctx: RowContext, *, own: bool) -> str:
    """:func:`friction_pair` of a contact with a spin row (``spin``): the pair and the spin row on their elliptic
    cone."""
    op, diag = ctx.operator, ctx.diagonal
    mu, spin_mu = ctx.mu(ctx.row), "spin_mu"
    block = (
        f"wp::mat33({diag.value(ctx.row)}, cross, spin_cross, cross, {diag.value('sib')}, spin_cross_sibling,\n"
        f"              spin_cross, spin_cross_sibling, {diag.value('spin')})"
    )
    lines = [
        f"const float spin_impulse = {ctx.load('spin')};",
        f"const float spin_mu = {ctx.mu('spin')};",
        "wp::mat33 triple_block(0.0f);",
        "wp::vec3 triple_residual(0.0f);",
        "// Zero load gives a zero cone; avoid unused reductions.",
        "if (load > 0.0f) {",
        _pair_terms(ctx, own=own),
        _spin_terms(ctx),
        f"    row_bound = {_row_bound(ctx, op.terms('residual', ctx.rhs(ctx.row), whole=True))};" if own else "",
        f"    sib_bound = {_sibling_bound(ctx)};",
        f"    spin_bound = {_spin_bound(ctx)};",
        f"    triple_block = {block};",
        "    triple_residual = wp::vec3(residual, sibling_residual, spin_residual);",
        "}",
        f"const wp::vec3 triple = {ctx.triple_law}triple_block, triple_residual,",
        f"    wp::vec3(old_impulse, sibling_impulse, spin_impulse), {mu}, {spin_mu}, load, omega);",
        "new_impulse = triple[0];",
        "sib_new = triple[1];",
        "sib_delta = sib_new - sibling_impulse;",
        "spin_new = triple[2];",
        "spin_delta = spin_new - spin_impulse;",
    ]
    return "\n".join(line for line in lines if line != "")


# ---------------------------------------------------------------------------------------------------------------------
# The row visit: a row loop's admission of row ``i`` before the row law
# ---------------------------------------------------------------------------------------------------------------------


def admit_pair(
    kind: str, parent: str, pad: str, *, disk: str = "", torsion: bool = False, spin: str | None = None
) -> str:
    """Friction rows of the row loop over ``i``: visit the first tangent only.

    Declares ``parent`` (the contact's normal row ``parent`` for a friction row, -1 for other kinds) and ``radius``
    (with ``torsion`` also the pooled normal ``load`` and the contact's spin row ``spin``, the expression ``spin`` of
    ``parent``: :func:`spin_row`), which :func:`friction_radius` sets, or ``disk`` (:func:`pair_radius`'s statements,
    indented one level) in the same block.
    """
    if torsion and spin is None:
        raise ValueError("a sweep with spin rows reads each contact's rows (spin_row)")
    lines = [
        "int parent = -1;",
        "int spin = -1;" if torsion else "",
        "float radius = 0.0f, load = 0.0f;" if torsion else "float radius = 0.0f;",
        f"if ({kind} == {FRICTION}) {{",
        f"    parent = {parent};",
        "    if (i != parent + 1) continue;",
        f"    spin = {spin};" if torsion else "",
        disk,
        "}",
    ]
    return indent("\n".join(line for line in lines if line), pad)


def pair_radius(
    ring_load: str,
    mu: str,
    load_row: str,
    load_next: str,
    pad: str,
    *,
    keep_next: str | None = None,
    load_spin: str | None = None,
) -> str:
    """Set ``radius``: ``mu`` times the pooled normal load of the contact's patch ring, and skip a zero pair.

    A zero disk with no carried tangent impulse (``load_row``, ``load_next``) cannot change the velocity, so the
    pair is skipped before its residuals are reduced. ``keep_next`` names a declared register that keeps the second
    tangent's impulse for the pair's update. With the spin row's impulse ``load_spin`` (torsion) the pooled ``load``
    is set too, and a zero cone with no carried impulse is skipped: the spin's capacity holds with a zero ``mu``.
    """
    lines = [f"const float lambda_n = {ring_load};", f"radius = fmaxf({mu} * lambda_n, 0.0f);"]
    if load_spin is not None:
        lines.append("load = fmaxf(lambda_n, 0.0f);")
    if keep_next is not None:
        lines.append(f"{keep_next} = {load_next};")
        load_next = keep_next
    if load_spin is not None:
        lines.append(f"if (load == 0.0f && {load_row} == 0.0f && {load_next} == 0.0f && {load_spin} == 0.0f) continue;")
    else:
        lines.append(f"if (radius == 0.0f && {load_row} == 0.0f && {load_next} == 0.0f) continue;")
    return indent("\n".join(lines), pad)


def friction_radius(
    kind: str,
    ring_load: str,
    mu: str,
    load_row: str,
    load_next: str,
    pad: str,
    *,
    keep_next: str | None = None,
    load_spin: str | None = None,
) -> str:
    """:func:`pair_radius` for a friction row of the row loop (one of kind ``kind``)."""
    radius = pair_radius(ring_load, mu, load_row, load_next, pad + "    ", keep_next=keep_next, load_spin=load_spin)
    return f"{pad}if ({kind} == {FRICTION}) {{\n{radius}\n{pad}}}"


def row_sums(op, out: str, pad: str, *, row: str = "i", sibling: str | None = None, paired: str = "false") -> str:
    """``float out = Z_row . u``, the visited row's velocity term (every lane holds it), by the operator's ``dot``.

    With ``sibling`` a ``paired`` row (a first tangent) also reduces its pair's terms in the same ladder,
    ``sibling_jv = Z_sibling . u`` and ``pair_coupling = Z_row . Z_sibling``, on the support's one load (rows in
    memory: :attr:`.ResponseOperator.pair_with_visit`), the sibling's entries kept for the pair's apply.
    """
    if sibling is None:
        return op.dot(((out, "row", None),), {"row": row}, pad)
    sums = ((out, "row", None), ("sibling_jv", "sib", None), ("pair_coupling", "row", "sib"))
    return op.dot(sums, {"row": row, "sib": sibling}, pad, exists={"sib": paired})


def row_visit(
    ctx: RowContext,
    laws: RowLaws,
    *,
    parent: str,
    check: str | None,
    declare: str = "",
    hold: str = "",
    keep_next: str | None = None,
    pad: str = "            ",
) -> str:
    """Visit row ``i`` (``ctx.row``) of a row loop, its kind ``ctx.kind`` declared: admission, residual and row law.

    A pair is visited at its first tangent (``parent``: the row's parent, :func:`admit_pair`) and skipped while its disk and
    impulses are zero (:func:`friction_radius`), before its residual is reduced. Every other row needs a positive
    diagonal ``check`` (its value or stored inverse, declared by ``declare``), unless the schedule admitted the rows
    on it already (``None``). ``hold`` declares, after the check, registers the disk and the law read (``keep_next``
    keeps the second tangent's impulse in one); the disk joins the admission's block when nothing comes between
    them. The residual's velocity term ``jv`` is the operator's (:func:`row_sums`); where the rows have friction pairs
    and the operator's rows are in memory (:attr:`.ResponseOperator.pair_with_visit`), a first tangent's ladder also
    reduces its pair's terms, on the support's one load. Rows without friction pairs (``ctx.friction`` false) have no
    admission or disk. A skipped row continues the loop.
    """
    row, kind, op = ctx.row, ctx.kind, ctx.operator
    pairs = ctx.friction
    between = [indent(declare, pad) if declare else ""]
    if check is not None:
        exempt = f" && {kind} != {FRICTION}" if pairs else ""
        between.append(f"{pad}if ({check} <= 0.0f{exempt}) continue;")
    between.append(indent(hold, pad) if hold else "")
    lines = []
    disk = (ctx.ring_load("parent"), ctx.mu(row), ctx.load(row), ctx.load(f"{row} + 1"))
    # Under torsion the contact's spin row, if any, follows the pair (:func:`spin_row`).
    spin = {"load_spin": f"(spin >= 0 ? {ctx.load('spin')} : 0.0f)"} if laws.torsion else {}
    admit = {"torsion": True, "spin": ctx.spin_row("parent")} if laws.torsion else {}
    if pairs and not any(between):
        disk_lines = pair_radius(*disk, "    ", keep_next=keep_next, **spin)
        lines.append(admit_pair(kind, parent, pad, disk=disk_lines, **admit))
    else:
        lines += [
            admit_pair(kind, parent, pad, **admit) if pairs else "",
            *between,
            friction_radius(kind, *disk, pad, keep_next=keep_next, **spin) if pairs else "",
        ]
    pair = {"sibling": "parent + 2", "paired": f"{kind} == {FRICTION}"} if pairs and op.pair_with_visit else {}
    lines += [row_sums(op, "jv", pad, **pair), row_update(ctx, laws, residual="jv", pad=pad)]
    return "\n".join(line for line in lines if line)


# ---------------------------------------------------------------------------------------------------------------------
# The contact unit: a contact normal and its friction pair on the unit's Gram block
# ---------------------------------------------------------------------------------------------------------------------


# The unit's rows by their names in the visit: the visited row (the normal, then the first tangent), its friction
# sibling and its spin row. A row's velocity term is its reduced term moved by the normal's change.
_UNIT_MOVED = {
    "row": "r == 0 ? unit_jv0 : unit_jv1 + gram_n1 * unit_dn",
    "sib": "(unit_jv2 + gram_n2 * unit_dn)",
    "spin": "(unit_jv3 + gram_n3 * unit_dn)",
}
_UNIT_GRAM = {("row", "sib"): "unit_cross", ("row", "spin"): "unit_cross13", ("sib", "spin"): "unit_cross23"}
_UNIT_CHANGE = {
    "row": "if (r == 0) unit_dn += {d}; else unit_dt1 += {d};",
    "sib": "unit_dt2 += {d};",
    "spin": "unit_ds += {d};",
}


@dataclasses.dataclass(frozen=True)
class UnitZ(ResponseOperator):
    """A contact unit's rows (:func:`contact_unit`) on the unit's Gram block: every placement's one operator for them.

    A contact unit is a contact normal (the unit's row ``r == 0``) and its friction pair (``r == 1``: the first tangent,
    the second its sibling), visited by the row law (:func:`row_visit`) one after the other on the unit's Gram block
    ``G = Z^T Z``. The placement reduces the unit's velocity terms once, before its rows (``unit_jv0``, ``unit_jv1``,
    ``unit_jv2``), and gives the pair's Gram couplings with the normal (``gram_n1``, ``gram_n2``): a cache formed once
    per launch, the builder's record, or the same reduction. The rows read the pair's terms moved by the normal's
    change, ``Z_t^T (u + Z_n dn) = Z_t^T u + G_tn dn``, and defer their applies into ``unit_dn``, ``unit_dt1`` and
    ``unit_dt2``, which the placement applies once after the unit: in exact arithmetic the row-by-row sweep.

    The friction rows' couplings (``Z_t1 . Z_t2``, and the spin row's with the pair) are the placement's reduction's
    (``unit_cross``, ``unit_cross13``, ``unit_cross23``: ``cached`` false), or none where the friction law reads a
    metric cached for the unit (``cached``). A row's vote reads its terms' magnitude through a lower bound
    (:meth:`terms`): the reduced term's own, or, where the placement's ladder also reduced the magnitudes
    (``magnitudes``: ``unit_jv{k}_terms``, and ``C_t = sum_k |Z_tk Z_nk|`` in ``gram_n{t}_terms``), the normal's
    ``S_n`` and a moved tangent's ``max(|Z_t^T u'|, S_t - |dn| C_t)``, the row-by-row sweep's ``S_t`` itself when the
    normal did not move. ``fused_pair_apply`` is the placement's tail (:func:`_row_tail`).
    """

    cached: bool = False
    magnitudes: bool = False
    fused_pair_apply: bool = False

    def terms(self, out: str, rhs: str, *, whole: bool) -> str:
        if not self.magnitudes:
            return super().terms(out, rhs, whole=whole)
        moved = terms_bound(out, rhs) if whole else f"fabsf({out})"
        if out == "sibling_residual":
            return f"fmaxf({moved}, unit_jv2_terms - fabsf(unit_dn) * gram_n2_terms)"
        if out == "spin_residual":
            return f"fmaxf({moved}, unit_jv3_terms - fabsf(unit_dn) * gram_n3_terms)"
        return f"(r == 0 ? unit_jv0_terms : fmaxf({moved}, unit_jv1_terms - fabsf(unit_dn) * gram_n1_terms))"

    def dot(self, sums, rows: dict[str, str], indent: str, *, rhs: dict[str, str] | None = None, **_) -> str:
        """``float out = Z_row . other`` for each ``(out, row, other)`` of ``sums`` on the unit's block: a velocity
        term the placement reduced, moved by the normal's change (``Z_t^T (u + Z_n dn)``), its right-hand side
        ``rhs`` added in the same expression; a Gram entry the placement's reduction gave, or, where the friction law
        reads a metric cached for the unit (``cached``), zero, the Gram entries of one call declared together. The
        operator's ladder and entries are the placement's, so the velocity form's other options read nothing here."""
        rhs = rhs or {}
        lines, grams = [], []
        for out, row, other in sums:
            if other is not None:
                grams.append((out, _UNIT_GRAM[(row, other)]))
                continue
            if row == "row" and out in rhs:
                raise ValueError("a contact unit's first tangent reads its moved residual through the row visit")
            added = f" + {rhs[out]}" if out in rhs else ""
            lines.append(f"{indent}float {out} = {_UNIT_MOVED[row]}{added};")
        if grams and self.cached:
            replaced = "the coupling" if len(grams) == 1 else "them"
            zeros = ", ".join(f"{out} = 0.0f" for out, _ in grams)
            lines.append(f"{indent}const float {zeros};  // the cached metric replaces {replaced}")
        else:
            lines += [f"{indent}const float {out} = {value};" for out, value in grams]
        return "\n".join(lines)

    def axpy(self, changes: tuple[tuple[str, str], ...], indent: str, *, row: str | None = None, **_) -> str:
        """``u += Z_k d_k`` of each ``(name, d)`` of ``changes``, deferred into the unit's changes (``unit_dn``,
        ``unit_dt1``, ``unit_dt2``, ``unit_ds``), which the placement applies once after the unit; ``row`` names the
        visit's row of one change."""
        return "".join(f"\n{indent}" + _UNIT_CHANGE[row or name].format(d=delta) for name, delta in changes)


def cached_pair_law(name: str, low: str, high: str, axis: tuple[str, str], pad: str) -> str:
    """Declare ``name``, the friction pair's law (:attr:`RowContext.pair_law`'s arguments after one leading ``int``)
    from a metric cached for the unit: the two eigenvalues ``low`` and ``high`` and the larger's ``axis``
    (:func:`.projection_laws.friction_pair_from_metric`); the visit's diagonals and coupling go unread."""
    lines = [
        f"const auto {name} = [&](int, float, float, float, wp::vec2 residual, wp::vec2 old,",
        f"{' ' * (len(name) + 18)}float radius, float relaxation) -> wp::vec2 {{",
        f"    return ProjectionLaw::friction_pair_from_metric({low}, {high},",
        f"        wp::vec2({axis[0]}, {axis[1]}), residual, old, radius, relaxation);",
        "};",
    ]
    return "\n".join(pad + line for line in lines)


def cached_triple_law(
    name: str, pad: str, *, metric: tuple[str, ...] = (), record: str = "", basis: bool = False
) -> str:
    """Declare ``name``, the friction triple's law (:attr:`RowContext.triple_law`'s arguments after one leading
    ``int``, the contact's normal row ``row``) from a metric cached for the unit, eight values
    (:func:`.projection_laws.friction_ball_metric`): its three eigenvalues and an unused value, then its eigenbasis as
    a rotation quaternion ``(x, y, z, w)`` (:func:`.projection_laws.friction_triple_from_metric`); the visit's block
    goes unread. The values are ``metric``'s expressions, or two ``float4`` loads from ``record`` (an address
    expression of ``row``). With ``basis`` the record holds the eigenbasis itself (:data:`FRICTION_METRIC_FLOATS`), four
    ``float4``, marked unformed before the sweeps (a negative first eigenvalue, :func:`friction_metric_stage`): the
    launch's first loaded visit forms it from its block and stores it, and a zero spin coefficient takes the pair's
    law from the block, so the law is the visit's own arithmetic (:func:`.projection_laws.friction_triple`)."""
    loads = []
    block = ""
    if record and basis:
        block = " block"
        loads = [
            # Without a spin coefficient the triple is its pair, by the pair's law (projection_laws.friction_ball).
            "    if (mu_t <= 0.0f) {",
            "        wp::vec3 pair(0.0f);",
            "        if (load > 0.0f) {",
            "            const wp::vec2 tangents = ProjectionLaw::friction_pair(block.data[0][0], block.data[0][1],",
            "                block.data[1][1], wp::vec2(residual[0], residual[1]), wp::vec2(old[0], old[1]), mu * load,",
            "                relaxation);",
            "            pair = wp::vec3(tangents[0], tangents[1], 0.0f);",
            "        }",
            "        return ConeLaw::ball(pair, mu, mu_t, load);",
            "    }",
            f"    float4* const record = reinterpret_cast<float4*>({record});",
            "    float4 metric = record[0];",
            "    wp::mat33 vectors;",
            "    if (metric.x < 0.0f && load > 0.0f) {",
            "        // The launch's first loaded visit forms the metric; every lane stores the same values.",
            "        const auto formed = ProjectionLaw::friction_ball_metric(block, mu, mu_t);",
            "        vectors = formed.v1;",
            "        float* const stored = reinterpret_cast<float*>(record);",
            "        for (int f = 0; f < 3; ++f) stored[f] = formed.v0[f];",
            "        for (int f = 0; f < 9; ++f) stored[4 + f] = vectors.data[f / 3][f % 3];",
            "        metric = record[0];",
            "    } else {",
            "        const float4 b0 = record[1], b1 = record[2], b2 = record[3];",
            "        vectors = wp::mat33(b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w, b2.x);",
            "    }",
        ]
        vectors = "vectors"
        metric = ("metric.x", "metric.y", "metric.z", "metric.w")
    else:
        if record:
            loads = [
                f"    const float* record = {record};",
                "    const float4 metric = *reinterpret_cast<const float4*>(record);",
                "    const float4 basis = *reinterpret_cast<const float4*>(record + 4);",
            ]
            metric = (*(f"metric.{c}" for c in "xyzw"), *(f"basis.{c}" for c in "xyzw"))
        x, y, z, w = metric[4:]
        vectors = f"wp::quat_to_matrix(wp::quat({x}, {y}, {z}, {w}))"
    v0, v1, v2 = metric[:3]
    lines = [
        f"const auto {name} = [&](int row, const wp::mat33&{block}, wp::vec3 residual, wp::vec3 old, float mu, float mu_t,",
        f"{' ' * (len(name) + 18)}float load, float relaxation) -> wp::vec3 {{",
        *loads,
        f"    return ConeLaw::friction_triple_from_metric(wp::vec3({v0}, {v1}, {v2}),",
        f"        {vectors}, residual, old, mu, mu_t, load, relaxation);",
        "};",
    ]
    return "\n".join(pad + line for line in lines)


UNIT_HELD = ("unit_lam", "unit_rhs", "unit_diag")
"""The registers in which a contact unit's placement holds its rows' staged values (:func:`contact_unit`), numbered by
the unit's row (0 the normal, 1 and 2 its pair, 3 its spin row under contact torsion): the impulses, right-hand sides
and diagonals."""


def unit_held_loads(
    ctx: RowContext, name: str, head: str, pad: str, *, torsion: bool = False, rows: str | None = None
) -> str:
    """Declare the held registers ``name`` (one of :data:`UNIT_HELD`) of the contact unit headed by row ``head``, read
    from the row store of ``ctx``: a placement's loads with the unit's rows, ahead of its reduction; under ``torsion``
    the spin row's fourth, read where the unit's ``rows`` hold one (:func:`spin_row`) and 0 otherwise."""
    read = dict(zip(UNIT_HELD, (ctx.load, ctx.rhs, ctx.diagonal.diagonal), strict=True))[name]
    values = [read(row) for row in (head, f"{head} + 1", f"{head} + 2")]
    if torsion:
        if rows is None:
            raise ValueError("a unit with spin rows holds its fourth row where its rows hold one")
        values.append(f"({rows}) > 3 ? {read(f'{head} + 3')} : 0.0f")
    return f"{pad}const float " + ", ".join(f"{name}{k} = {value}" for k, value in enumerate(values)) + ";"


def contact_unit(
    ctx: RowContext,
    laws: RowLaws,
    *,
    rows: str,
    declare: str,
    parent: str,
    check: str | None,
    held: bool = False,
    pad: str,
) -> str:
    """The rows of a contact unit by the row law (BACKBONE §1 step 5), after the placement's one reduction of the
    unit's terms and before its one apply of the unit's changes (both the placement's).

    Row ``r`` of the unit (the normal, then its pair at the first tangent; ``rows`` is the unit's row count, one for a
    normal without a pair, else three) declares its index ``i``, kind and response (``declare``, which ``check`` reads:
    the row law runs when it is positive; ``None`` where the placement admitted the unit's normal) and runs
    :func:`row_visit` with the unit's operator (:class:`UnitZ`), the pair at the normal ``parent``. A visit that
    skips its row leaves it (a one-pass loop), not the unit.

    The rows defer their applies, so none synchronizes the lanes after it: the placement does once, after the unit's
    apply. Where the rows' first lane owns the stores, the placement holds the unit's staged values in registers
    (``held``: :data:`UNIT_HELD`, the impulses, right-hand sides and diagonals of the normal and its two tangents,
    loaded with its rows ahead of its reduction, :func:`unit_held_loads`), which the rows read, and the pair pools its
    ring from the normal's new impulse in a register (``unit_normal``, what the normal stores). The rows then read no
    store of the unit, so only the unit's first stores wait for every lane's reads: the normal's where the placement
    admitted it, else each row's (a skipped normal leaves the pair's stores first). Nor does a row read its staged
    values after such a barrier, which orders the reads after it (a memory fence): the pair's right-hand sides and
    diagonals would load after the normal's stores, on the unit's critical chain. Rows that every lane stores alike, or
    one thread's, pass no barrier and read their own stores.

    Where the laws allow spin rows (``laws.torsion``) a unit of four ``rows`` has its spin row after the pair, which
    the pair's visit solves with it (the friction triple, :func:`friction_pair`); a unit of three solves its pair as
    the triple of a zero spin coefficient (:func:`spin_row`). ``held`` holds the spin row's staged values fourth, its
    change defers into ``unit_ds``, and the placement reduces its terms (``unit_jv3``, ``gram_n3``, ``unit_cross13``,
    ``unit_cross23``: :func:`_spin_terms`, zero for a unit of three) and applies it with the unit's.
    """
    if (held and not ctx.owned) or (ctx.lanes.parallel and ctx.owned and not held):
        raise ValueError(
            "a contact unit holds its staged values in registers exactly where its first lane owns the stores (one "
            "thread may hold them: it reads them as its stores)"
        )
    unit = dataclasses.replace(ctx, synced=False, first_store="r == 0" if check is None else None)
    if laws.torsion:
        # The unit's rows are the data's: a fourth row is the pair's spin row (:func:`spin_row`).
        unit = dataclasses.replace(unit, spin_row=lambda contact: f"(({rows}) > 3 ? ({contact}) + 3 : -1)")
    spin = ", unit_ds = 0.0f" if laws.torsion else ""
    declarations = [f"{pad}float unit_dn = 0.0f, unit_dt1 = 0.0f, unit_dt2 = 0.0f{spin};"]
    if held:
        if ctx.diagonal.diagonal is None:
            raise ValueError("a held contact unit reads its rows' diagonals")
        impulse, rhs, diagonal = UNIT_HELD

        def held_row(name: str, row: str) -> str | None:
            # The visited row (the normal, then the first tangent), the row after it and the pair's second tangent;
            # under torsion the row two after the visited one and the spin row.
            index = {ctx.row: (0, 1), f"{ctx.row} + 1": (1, 2), "sib": (2, 2)}
            if laws.torsion:
                index.update({f"{ctx.row} + 2": (2, 3), "spin": (3, 3)})
            if row not in index:
                return None
            normal, pair = index[row]
            return f"{name}{pair}" if normal == pair else f"(r == 0 ? {name}{normal} : {name}{pair})"

        def store(row: str, value: str) -> str:
            stored = ctx.store(row, value)
            return f"{stored}\nif (r == 0) unit_normal = {value};" if row == ctx.row else stored

        unit = dataclasses.replace(
            unit,
            diagonal=dataclasses.replace(
                ctx.diagonal, diagonal=lambda row: held_row(diagonal, row) or ctx.diagonal.diagonal(row)
            ),
            load=lambda row: held_row(impulse, row) or ctx.load(row),
            rhs=lambda row: held_row(rhs, row) or ctx.rhs(row),
            store=store,
            ring_load=lambda row: ctx.ring_load(row, "unit_normal"),
        )
        declarations.append(f"{pad}float unit_normal = {impulse}0;")
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
    violation: tuple[str, str, str, str] | None = None,
    block: bool = False,
    iterations: str = "iterations",
    argmax: str | None = "ARGMAX",
) -> str:
    """The end of a sweep under the exit law (:func:`.solve_policy.sweep_exit`): stop once no update was loud.

    ``violation`` bounds
    the rows a sweep of selected rows leaves unvisited. A ``block`` group spans several warps: its quiet test is the
    block's vote (:data:`BLOCK_QUIET`) and ``argmax`` its trace reduction (:func:`.solve_policy.block_argmax`).
    ``iterations`` is the sweep budget.
    """
    quiet = BLOCK_QUIET if block else QUIET
    return sweep_exit(
        policy,
        quiet=quiet,
        violation=violation,
        iteration=iteration,
        iterations=iterations,
        argmax=argmax,
    )


# ---------------------------------------------------------------------------------------------------------------------
# Schedules: how a phase enumerates its units, and the sweeps' iteration frame
# ---------------------------------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class SlotOrder:
    """A phase's rows ``[0, end)`` in slot order, row ``i`` at a time (the matrix-free sweep's dense rows, the sparse
    sweep without its active subword, the sparse-diagonal sweep's coupled rows)."""

    end: str

    def inner(self, pad: str) -> str:
        """The visit's indentation inside the loop at ``pad``."""
        return pad + "    "

    def wrap(self, visit: str, pad: str) -> str:
        return f"{pad}for (int i = 0; i < {self.end}; ++i) {{\n{visit}\n{pad}}}"


@dataclasses.dataclass(frozen=True)
class SelectedRows:
    """The rows a scan at the sweep's entry selected into the bitset ``words`` (the sparse sweep's active subword,
    :func:`.sweep_routes_sparse.active_scan`), popped in slot order, row ``i`` at a time."""

    count: str
    words: str = "s_selected_words"

    def inner(self, pad: str) -> str:
        return pad + "        "

    def wrap(self, visit: str, pad: str) -> str:
        return (
            f"{pad}for (int word = 0; word < ({self.count} + 31) / 32; ++word) {{\n"
            f"{pad}    unsigned pending = {self.words}[word];\n"
            f"{pad}    while (pending != 0u) {{\n"
            f"{pad}        const int i = word * 32 + FIRST(pending);\n"
            f"{pad}        pending &= pending - 1u;\n"
            f"{visit}\n"
            f"{pad}    }}\n"
            f"{pad}}}"
        )


@dataclasses.dataclass(frozen=True)
class Pipelined:
    """A phase's rows ``[0, end)`` in slot order with what a row's visit reads first loaded a row ahead: ``prologue``
    loads row 0's before the loop, ``consume(row, indent)`` takes row ``row``'s and loads the next's (the free
    bodies' header words and ``Z`` entries, :func:`rows_block`; the paired factor sweep's factor rows)."""

    end: str
    prologue: Callable[[str], str]
    consume: Callable[[str, str], str]

    def inner(self, pad: str) -> str:
        return pad + "    "

    def wrap(self, visit: str, pad: str) -> str:
        return (
            f"{self.prologue(pad)}\n{pad}for (int i = 0; i < {self.end}; ++i) {{\n"
            f"{self.consume('i', pad + '    ')}\n{visit}\n{pad}}}"
        )


@dataclasses.dataclass(frozen=True)
class BallotHeads:
    """The heads of a phase's units, found by the lanes together, a lane a row of ``[0, count)`` (``test(row)``),
    and visited in slot order, ``head`` at a time (the sparse sweep's friction triples, :func:`.sweep_routes_sparse.
    triple_phase`); ``guard`` skips the phase where the world holds none."""

    count: str
    lanes: Lanes
    test: Callable[[str], str]
    guard: str = ""

    def inner(self, pad: str) -> str:
        return pad + "        "

    def wrap(self, visit: str, pad: str) -> str:
        guard = f"if ({self.guard}) " if self.guard else ""
        return (
            f"{pad}{guard}for (int first = 0; first < {self.count}; first += {self.lanes.width}) {{\n"
            f"{pad}    const int row = first + {self.lanes.lane};\n"
            f"{pad}    unsigned heads = BALLOT({self.test('row')});\n"
            f"{pad}    while (heads != 0u) {{\n"
            f"{pad}        const int head = first + FIRST(heads);\n"
            f"{pad}        heads &= heads - 1u;\n"
            f"{visit}\n"
            f"{pad}    }}\n"
            f"{pad}}}"
        )


@dataclasses.dataclass(frozen=True)
class UnitStride:
    """A phase's rows ``[0, end)`` in slot order a unit at a time: the visit sets ``step`` to its unit's rows (one, or a
    contact's three or four: the paired factor sweep's contact units)."""

    end: str

    def inner(self, pad: str) -> str:
        return pad + "    "

    def wrap(self, visit: str, pad: str) -> str:
        return f"{pad}for (int i = 0, step = 1; i < {self.end}; i += step) {{\n{visit}\n{pad}}}"


ITERATION = "iter"
"""The sweep counter of :func:`level_loop`, as the exit law reads it (:func:`end_sweep`'s ``iteration``)."""


def level_loop(
    phases: Sequence[str],
    last_word: Sequence[str] = (),
    *,
    exit: str,
    iteration: str = ITERATION,
    budget: str = "iterations",
    resume: bool = False,
    pad: str = "    ",
) -> str:
    """The iteration frame of every sweep: per sweep, no change voted yet (``iteration_changed``), the schedule's
    phases in order, the last word (the rows sorted last: velocity limits, the free bodies' speed rows), then the exit
    law (``exit``: :func:`end_sweep` at ``iteration``), which ends the loop once a sweep is quiet, or the ``budget``
    of sweeps. With ``resume`` the counter is the caller's and the loop takes it up where it stopped: a Jacobi law's
    epochs, each a run of sweeps of one metric (:func:`jacobi_sweeps`)."""
    body = "\n".join(text.rstrip("\n") for text in (*phases, *last_word) if text.strip())
    start = "" if resume else f"int {iteration} = 0"
    return (
        f"{pad}for ({start}; {iteration} < {budget}; ++{iteration}) {{\n"
        f"{pad}    int iteration_changed = 0;\n"
        f"{body}\n{exit.rstrip(chr(10))}\n{pad}}}"
    )


# ---------------------------------------------------------------------------------------------------------------------
# Row blocks: a row store, an operator and the row law, over a range of rows in slot order
# ---------------------------------------------------------------------------------------------------------------------


FRICTION_METRIC_FLOATS = 16
"""A friction triple's metric, cached for a launch (:func:`friction_metric_stage`): its block's three eigenvalues and
an unused value, then its eigenbasis's nine entries row by row (:func:`.projection_laws.friction_ball_metric`), three
floats of padding: four ``float4`` (:func:`cached_triple_law` with ``basis``). The basis itself, not the quaternion the
support placement's records and the colored units' slots round it through, so the cached law is the visit's own
arithmetic."""


def friction_metric(row: str, first: str = "") -> str:
    """The address of the friction metric of the triple whose normal is row ``row`` of the world's line (from
    ``first``, the class's first row in the line): ``friction_metric``'s slot ``row / 4`` of line ``line``. A triple
    holds four rows, so two triples never share a slot."""
    at = f"({first} + ({row}))" if first else f"({row})"
    return (
        f"(friction_metric.data + (static_cast<size_t>(line) * friction_metric.shape[1] + {at} / 4)"
        f" * {FRICTION_METRIC_FLOATS})"
    )


def friction_metric_stage(
    kind: Callable[[str], str],
    parent: Callable[[str], str],
    spin: Callable[[str], str],
    *,
    end: str,
    record: Callable[[str], str],
    pad: str,
) -> str:
    """Before the sweeps, mark the metric of every friction triple (a contact with a spin row ``spin(normal)``,
    :func:`spin_row`) of a class's rows ``[0, end)`` unformed (a negative first eigenvalue at ``record(parent)``), each
    row's ``kind`` and ``parent`` as the row store reads them: the lanes find the triples together, a lane a row. A
    triple's first loaded visit of the launch forms its metric and stores it, and every later visit only projects
    (:func:`cached_triple_law` with ``basis``): its block is fixed across the launch's sweeps.

    Formed at the first visit, not here: a triple the sweeps never load forms none."""
    lines = [
        "const int row = first + LANE;",
        f"if (row < {end} && {kind('row')} == {FRICTION}) {{",
        f"    const int parent = {parent('row')};",
        f"    if (row == parent + 1 && {spin('parent')} >= 0) {record('parent')}[0] = -1.0f;",
        "}",
    ]
    text = f"for (int first = 0; first < {end}; first += STRIDE) " + _block(lines, "")
    return indent(text + "\nSYNC();", pad)


def _block(lines: list[str], pad: str) -> str:
    """``lines`` as a braced block whose closing brace sits at ``pad``."""
    return "{\n" + indent("\n".join(lines), pad + "    ") + f"\n{pad}}}"


def _record_load(rows: WorldRows, op, row: str, indent: str) -> str:
    """Load row ``row``'s record into the row loop's pipeline registers (``pre_a``, ``pre_b``), and where the operator
    keeps the visited row's entries in registers, this lane's entry of its ``Z`` (``pre_z``)."""
    text = rows.load_record(tuple(f"pre_{name[-1]}" for name in rows.registers), row, indent)
    if op.registers:
        text += f"\n{indent}if ({op.lanes.lane} < {op.width}) pre_z = {op.element(row, op.lanes.lane)};"
    return text


def _record_consume(rows: WorldRows, op, row: str, end: str, indent: str) -> str:
    """Take row ``row``'s record (``rec_a``, ``rec_b``) and entry (``cur_z``) from the pipeline and load row
    ``row + 1``'s (before ``end``); a free row's header word decodes into its sides' coordinates and its fields
    (:func:`.row_metadata.unpack_mf_header`)."""
    lines = [f"{indent}const int4 {name} = pre_{name[-1]};" for name in rows.registers]
    if op.registers:
        lines.append(f"{indent}float cur_z = pre_z;")
    lines += [
        f"{indent}if ({row} + 1 < {end}) {{",
        _record_load(rows, op, f"{row} + 1", indent + "    "),
        f"{indent}}}",
    ]
    if rows.free:
        lines.append(unpack_mf_header("rec_a", indent))
    return "\n".join(lines)


def _record_pipeline(rows: WorldRows, op, end: str) -> Pipelined:
    """A record store's rows ``[0, end)`` in slot order, each row's record loaded a row ahead."""
    z = "float pre_z = 0.0f;\n{pad}" if op.registers else ""
    return Pipelined(
        end,
        lambda pad: (
            f"{pad}int4 {', '.join(f'pre_{name[-1]}' for name in rows.registers)};\n{pad}{z.format(pad=pad)}"
            f"if ({end} > 0) {{\n{_record_load(rows, op, '0', pad + '    ')}\n{pad}}}"
        ),
        lambda row, pad: _record_consume(rows, op, row, end, pad),
    )


def row_step(
    rows: RowStore,
    op,
    laws: RowLaws,
    *,
    end: str,
    kind: int | None = None,
    parent: str | None = None,
    admission: str = "",
    loads: str = "",
    held: bool = False,
    triple_phase: bool = False,
    inline_velocity_limit: bool = False,
    metric: str | None = None,
    pad: str = "            ",
) -> str:
    """Row ``i``'s step in a row loop over a store's rows ``[0, end)``: its kind read from the store, the schedule's
    skips, then the shared visit (:func:`row_visit`): admission, residual and row law, in the store's context
    (:meth:`RowContext.of`: who owns a row's stores is the store's). Every row loop of the sparse, matrix-free,
    sparse-diagonal and paired factor sweeps runs this text; its data choices are all that differ:

    - ``held``: the row's and its pair's impulses are read at admission into registers (``row_impulse``,
      ``pair_impulse``) and the law reads them there (the sparse sweep's rows);
    - the store's reciprocal: a store that keeps the diagonals' reciprocals steps by multiplying, otherwise by
      dividing (:meth:`RowStore.diagonal`): a rounding difference kept as data;
    - ``triple_phase``: a friction triple's rows are solved in a phase of their own after the loop (the store restaged
      its pair as spin rows, :attr:`StagedRows.triples`), and the loop's visit carries none of the triple's code;
      otherwise a spin row is solved with its friction pair in the loop (the store's :meth:`RowStore.spin_of`), its
      metric cached for the launch with ``metric`` (the law :func:`friction_metric_law` declared,
      :func:`friction_metric_stage`);
    - ``inline_velocity_limit``: velocity-limit rows take their law in place, in slot order (the paired factor
      sweep), instead of the last word (:func:`velocity_limit_pass`), which ``admission`` then skips them for;
    - ``loads``: what the operator reads of the row and its pair in registers, loaded after the admission (the paired
      factor sweep's ``sibling_z`` and ``spin_z``);
    - a row known where the step is emitted (``kind``, and a friction row's ``parent``) replaces its metadata, so the
      step compiles to that kind's law: a known kind is no spin row, and one other than friction admits no pair (the
      sparse-diagonal sweep's chains).

    A spin row is skipped first (solved with its pair, here or in the triple phase), then ``admission``, the
    schedule's own skips (velocity limits, which the last word solves).
    """
    ctx = RowContext.of(
        rows,
        op,
        end=end,
        metric=metric,
        friction=kind in (None, FRICTION),
        inline_velocity_limit=inline_velocity_limit,
    )
    hold = {}
    if held:
        registers = {"i": "row_impulse", "sib": "pair_impulse"}
        ctx = dataclasses.replace(ctx, load=lambda row: registers.get(row) or rows.load(row))
        hold = {"hold": "const float row_impulse = " + rows.load("i") + ";\nfloat pair_impulse = 0.0f;"}
    head = [
        f"int row_type = {rows.kind('i') if kind is None else kind};",
        # A spin row is solved with its friction pair (torsion).
        f"if (row_type == {TORSION}) continue;" if laws.torsion and kind is None else "",
        admission,
    ]
    visit = row_visit(
        ctx,
        dataclasses.replace(laws, torsion=False) if triple_phase else laws,
        parent=rows.parent("i") if parent is None else parent,
        check="denom",
        declare=f"{loads}float denom = {rows.diag('i')};",
        keep_next="pair_impulse" if held else None,
        pad=pad,
        **hold,
    )
    return indent("\n".join(line for line in head if line), pad) + "\n" + visit


def rows_block(
    rows: WorldRows,
    op,
    laws: RowLaws,
    *,
    admission: str = "",
    end: str | None = None,
    metric: str | None = None,
    pad: str = "        ",
) -> str:
    """The Gauss-Seidel loop over a world's rows of one class ``[0, end)`` in slot order (:class:`WorldRows`): the
    dense rows on packed rows (:class:`.response_operator.SparseFactor`, to ``m_dense``), each by the row step
    (:func:`row_step`), the free bodies' on their ``Z`` (:func:`.sweep_routes.free_body_rows`, to
    ``mf_contact_end``). Where the rows' records stream, the loop is software-pipelined: each row's record (and, for
    the free bodies' operator, this lane's entry of its ``Z``) loads a row ahead (``pre_a``, ``pre_b``, ``pre_z``), and
    the visit takes them as ``rec_a``, ``rec_b`` and ``cur_z``.

    ``admission`` holds the block's own skips, run after the row's kind is read. A dense friction pair's second
    tangent shares its first tangent's entries: the first tangent's residual reduces the pair's terms with its own
    (:func:`row_sums`). A free row's impulses need no barrier around their stores: every lane stores
    the value it computed (or, streamed, the first lane stores and every read takes the first lane's value), so a lane
    only ever reads its own stores.

    With ``metric`` (the law :func:`friction_metric_law` declared) every friction triple forms its metric once per
    launch, at its first loaded visit (:func:`friction_metric_stage`).
    """
    inner = pad + "    "
    if end is None:
        end = "mf_contact_end" if rows.free else "m_dense"
    if not rows.free:
        visit = row_step(rows, op, laws, end=end, admission=admission, metric=metric, pad=inner)
        if rows.record:
            return f"{pad}{{\n{_record_pipeline(rows, op, end).wrap(visit, pad)}\n{pad}}}"
        return SlotOrder(end).wrap(visit, pad)
    cached = {} if metric is None else {"triple_law": f"{metric}(parent, "}
    ctx = RowContext(
        row="i",
        kind="mf_rt",
        operator=op,
        diagonal=rows.diagonal("mf_diag"),
        load=rows.load,
        store=rows.store,
        rhs=rows.rhs_of,
        mu=rows.mu,
        weight=rows.weight,
        parent="parent",
        ring_load=rows.ring_load,
        lanes=rows.lanes,
        eps=rows.eps,
        spin_row=lambda contact: rows.spin_of(contact, end),
        contact_kinds=(CONTACT,),
        owned=False,
        **cached,
    )
    body = [
        # A spin row is solved with its friction pair (torsion).
        f"{inner}if (mf_rt == {TORSION}) continue;" if laws.torsion else "",
        indent(admission, inner) if admission else "",
        row_visit(ctx, laws, parent=rows.parent_of("packed_tp"), check="mf_diag", declare="", pad=inner),
    ]
    return f"{pad}{{\n{_record_pipeline(rows, op, end).wrap(chr(10).join(line for line in body if line), pad)}\n{pad}}}"


def friction_triple_visit(
    ctx: RowContext, laws: RowLaws, *, head: str, metric: str, pool: str, load: str, law: str, pad: str
) -> str:
    """Visit the friction triple of the contact whose normal is row ``head``: its pair and its spin row
    (:func:`spin_row`), solved together on their elliptic cone (:func:`.projection_laws.friction_triple`) apart from
    the row loop, in a phase of their own after it.

    ``metric`` loads what the law and the pool read, first and off the chain; ``pool`` forms the pooled normal ``load``
    (an expression, its patch ring's), and a zero cone with no carried impulse is skipped (``continue``) before one
    reduction of the three rows' velocity terms on the contact's shared support (``dot``).
    The triple's law projects the residuals: ``law`` is the call up to its residual argument, ``(residual, old, mu,
    mu_t, load, omega)`` after it (:func:`.projection_laws.friction_triple_from_metric` of the contact's metric). The
    first lane stores the impulses, the three changes vote together and one apply writes each coordinate once
    (``axpy`` of a block): the three changes summed in the rows' order, then added.
    """
    op, diag = ctx.operator, ctx.diagonal
    names = (("tangent", "first"), ("sibling", "second"), ("spin", "spin"))
    rhs = {name: ctx.rhs(row) for row, name in names}
    changes = tuple(
        diag.change(f"{name}_delta", f"{name}_bound", str(TORSION if name == "spin" else FRICTION), row)
        for row, name in names
    )
    lines = [
        f"const int tangent = {head} + 1, sibling = {head} + 2, spin = {head} + 3;",
        metric,
        pool,
        f"const float load = fmaxf({load}, 0.0f);",
        *(f"const float {name}_old = {ctx.load(row)};" for row, name in names),
        "// Zero load gives a zero cone: a triple without impulse moves nothing.",
        "if (load == 0.0f && first_old == 0.0f && second_old == 0.0f && spin_old == 0.0f) continue;",
        op.dot(
            tuple((f"{name}_jv", name, None) for _, name in names),
            {name: row for row, name in names},
            "",
            keep_velocity=True,
        ),
        *(f"const float {name}_residual = {name}_jv + {rhs[name]};" for _, name in names),
        *(
            f"const float {name}_bound = {diag.bound(rhs[name], op.terms(f'{name}_jv', rhs[name], whole=False), f'{name}_old', row)};"
            for row, name in names
        ),
        f"const wp::vec3 triple = {law}wp::vec3(first_residual, second_residual, spin_residual),",
        f"    wp::vec3(first_old, second_old, spin_old), {ctx.mu('tangent')}, {ctx.mu('spin')}, load, omega);",
        *(f"const float {name}_delta = triple[{k}] - {name}_old;" for k, (_, name) in enumerate(names)),
        ctx.store_barrier(),
        *(ctx.store(row, f"triple[{k}]") for k, (row, _) in enumerate(names)),
        "if (first_delta != 0.0f || second_delta != 0.0f || spin_delta != 0.0f) {",
        change_vote(laws, changes, "    "),
        op.axpy(
            (("first", "first_delta"), ("second", "second_delta"), ("spin", "spin_delta")),
            "    ",
            kept=True,
            block=True,
        ),
        "}",
        ctx.sync(),
    ]
    return indent("\n".join(line for line in lines if line), pad)


def friction_metric_law(name: str, record: Callable[[str], str], pad: str) -> str:
    """Declare ``name``, the friction triple's law of a class's rows, its metric cached for the launch at
    ``record(row)`` (``row``: the contact's normal; :func:`friction_metric_stage`)."""
    return cached_triple_law(name, pad, record=record("row"), basis=True)


def impulse_seed(
    count: str,
    impulse: str,
    kind: str,
    apply: str,
    sync: str,
    *,
    skip: int = VELOCITY_LIMIT,
    declare: tuple[str, ...] = (),
    pad: str = "    ",
) -> str:
    """Give the velocity change the response of the impulses a sweep starts from, ``du = sum_i Z_i lambda_i``, row by
    row in order, before the first sweep: the warm start's and the closed loop's. A velocity-limit row carries no
    impulse into a sweep, and a free body's speed row no ``Z``: rows of kind ``skip`` are left out. A row's lanes'
    updates never collide; ``sync`` serializes the rows.

    Args:
        count: The rows.
        impulse: Row ``i``'s impulse.
        kind: Row ``i``'s type.
        skip: The kind left out (``VELOCITY_LIMIT`` or ``SPEED``).
        apply: ``du += Z_i seed`` (the operator's ``axpy`` of row ``i``, indented for this block).
        sync: The lanes' barrier after a row.
        declare: Statements row ``i``'s ``apply`` needs.
        pad: Indentation of the emitted statements.
    """
    lines = [
        f"for (int i = 0; i < {count}; ++i) {{",
        *(f"    {line}" for line in declare),
        f"    const float seed = {impulse};",
        f"    if (seed != 0.0f && {kind} != {skip}) {{",
        apply,
        "    }",
        f"    {sync}",
        "}",
    ]
    return indent("\n".join(lines), pad)


# ---------------------------------------------------------------------------------------------------------------------
# End-of-sweep passes: rows whose law gives them the last word of every sweep
# ---------------------------------------------------------------------------------------------------------------------


def velocity_limit_pass(rows: WorldRows, op, laws: RowLaws, *, end: str = "m_dense", pad: str = "            ") -> str:
    """Dense joint velocity-limit rows after every other row of the sweep, under the velocity-limit law.

    Each row keeps a unilateral total (:func:`.projection_laws.velocity_limit_update`), owned by the row's first
    lane. ``rows`` stages the rows' impulses, kinds, diagonals and right-hand sides (:class:`WorldRows`, or a route's
    store of the same accessors); the operator reduces each row's velocity term outside the row loop
    (``dot``).
    """
    lanes = rows.lanes
    visit = rows.visit("i")
    lines = [
        f"for (int i = 0; i < {end}; i++) {{",
        *((indent(visit, "    "),) if visit else ()),
        f"    if (({rows.kind('i', scan=True)}) != {VELOCITY_LIMIT}) continue;",
        f"    float denom = {rows.diag('i')};",
        "    if (denom <= 0.0f) continue;",
        "",
        op.dot((("my_sum_vlim", "row", None),), {"row": "i"}, "    "),
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
        op.axpy((("row", "delta_impulse_vlim"),), "        "),
        "    }",
        f"    {lanes.sync()}",
        "}",
    ]
    return indent("\n".join(lines), pad)


def speed_rows_pass(
    rows: RowStore,
    op,
    laws: RowLaws,
    *,
    walk,
    start: str = "mf_contact_end",
    end: str = "m_mf",
    body: tuple[str, str, str] | None = None,
    pad: str = "            ",
) -> str:
    """Free-body speed rows (``[start, end)``, a class's last rows) after every other row: each bounds the magnitude
    of its body's linear (axis ``e_0``) or angular (axis ``e_3``) velocity, along that velocity (:func:`speed_row`),
    which the body's ``walk`` reads and pushes; ``op`` holds the rows' entries and the velocity.

    The free bodies' run's rows (``rows.free``) carry their body's first coordinate, inverse diagonal and right-hand
    side in their record; the line's (a dense store) name their body by ``body``: the statements that
    declare row ``i``'s, the expression of its first world coordinate, and that of the entry its unwhitened axis marks
    at the body's coordinate 3 (:func:`.walk_rows.speed_axis`)."""
    lanes = rows.lanes
    visit = indent(rows.visit("i"), "    ") if hasattr(rows, "visit") else ""
    if getattr(rows, "free", False):
        lines = [
            f"for (int i = {start}; i < {end}; i++) {{",
            visit,
            unpack_mf_header("rec_a", "    "),
            f"    if (mf_rt != {SPEED}) continue;",
            "    if (mf_diag <= 0.0f || dof_a < 0) continue;",
            "",
            f"    const int speed_axis = {op.element('i', '3')} != 0.0f ? 3 : 0;",
        ]
        rhs, inverse, first = rows.rhs_of("i"), "mf_diag", "dof_a"
        store = lanes.first(rows.store("i", "speed_row[0]")) if not rows.stream else rows.store("i", "speed_row[0]")
    else:
        declare, first, axis = body
        lines = [
            f"for (int i = {start}; i < {end}; i++) {{",
            *((visit,) if visit else ()),
            f"    const float speed_diag = {rows.diag('i')};",
            "    if (speed_diag <= 0.0f) continue;",
            "    const float speed_inverse = 1.0f / speed_diag;",
            *(f"    {line}" for line in declare),
            f"    const int speed_axis = {axis} != 0.0f ? 3 : 0;",
        ]
        rhs, inverse = rows.rhs_of("i"), "speed_inverse"
        store = rows.store("i", "speed_row[0]")
    lines += [
        speed_row(
            walk,
            laws,
            velocity=op.velocity,
            load=rows.load("i"),
            store=store,
            lanes=lanes,
            rhs=rhs,
            inverse=inverse,
            body=first,
            pad="    ",
        ),
        "}",
    ]
    return indent("\n".join(lines), pad)


def speed_row(
    walk,
    laws: RowLaws,
    *,
    velocity: Velocity,
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

    The body's coordinates from ``body`` hold ``u``, its factor coordinates (``velocity``); the speed is the magnitude
    of its physical velocity there: every lane forms ``speed_v = L^-T u`` by the body's walk down
    (``walk.walk_leave(first, values)``) on the predictor, which ``v_out`` holds through the sweep, and the magnitude,
    the squares added in coordinate order; the lanes then synchronize. The row's law (:func:`.projection_laws.speed_row`)
    gives its stored value (``store`` stores ``speed_row[0]``, the total of the row's impulse ``load``) and the scale
    that pushes the velocity back along itself, ``u += L^T dv`` by the walk in (``walk.walk_enter(first, velocity,
    out)``), one thread or the first lane. Where each lane holds its own coordinate in a register
    (:class:`.response_operator.LaneVelocity`), the body's six lanes stage theirs in the world's ``s_speed`` first and
    take them back after. ``inverse`` is the row's inverse diagonal.
    """
    registers = isinstance(velocity, LaneVelocity)
    stage = take = ""
    if registers:
        held = f"{lanes.lane} >= {body} && {lanes.lane} < {body} + 6"
        stage = f"{pad}if ({held}) s_speed[{lanes.lane} - ({body})] = {velocity.array};\n{pad}{lanes.sync()}"
        take = f"{pad}if ({held}) {velocity.array} = s_speed[{lanes.lane} - ({body})];"
        v = "s_speed[k]"
    else:
        v = velocity.at(f"{body} + k")
    speed = [
        "float speed_v[6];",
        f"for (int k = 0; k < 6; ++k) speed_v[k] = {v};",
        walk.walk_leave(body, "speed_v"),
        f"for (int k = 0; k < 6; ++k) speed_v[k] += v_out.data[world_dof_indices.data[dof_map_base + {body} + k]];",
        f"const float speed_mf_vlim = sqrtf((speed_v[{axis}] * speed_v[{axis}] + speed_v[{axis} + 1] * speed_v[{axis} + 1])"
        f" + speed_v[{axis} + 2] * speed_v[{axis} + 2]);",
    ]
    if lanes.parallel:
        speed.append(lanes.sync())
    push = "\n".join(
        [
            "float speed_dv[6];",
            f"for (int k = 0; k < 6; ++k) speed_dv[k] = k >= {axis} && k < {axis} + 3 ? speed_row[2] * speed_v[k] : 0.0f;",
            "float speed_du[6];",
            walk.walk_enter(body, "speed_dv", "speed_du"),
            f"for (int k = 0; k < 6; ++k) {v} += speed_du[k];",
        ]
    )
    if lanes.parallel:
        push = f"if ({lanes.lane} == 0) {{\n{push}\n}}"
    lines = [
        *((stage,) if stage else ()),
        "\n".join(pad + line for line in "\n".join(speed).split("\n")),
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
            (Change("speed_row[1]", "bound_speed", str(SPEED), f"1.0f / {inverse}"),),
            pad + "    ",
        ),
        "\n".join(pad + "    " + line for line in push.split("\n")),
        f"{pad}}}",
    ]
    if lanes.sync():
        lines.append(f"{pad}{lanes.sync()}")
    if take:
        lines.append(take)
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------------------------------
# The dense-row tiers' worlds: a block per world without free-body rows, its dense rows within the tier's bounds
# ---------------------------------------------------------------------------------------------------------------------


def dense_tier_world(
    dofs: int,
    max_constraints: int,
    *,
    rows: str,
    support: int,
    pitch: int,
    pad: str = "    ",
) -> str:
    """Enter a world of a dense-row tier: one of its bucket of the claims' worlds (:func:`.sweep_levels.claim_worlds`),
    so one without free-body rows and with the tier's count of dense rows.

    Declares ``m_dense``, ``rows`` (the rows the tier holds, ``m_dense``), ``dof_map_base``, ``off_dense`` (the world's
    line of the row store, ``pitch`` rows a line), and the world's first packed row entry ``packed_base`` (the line's
    rows of ``support`` entries: :mod:`.packed_rows`).
    """
    base = f"const size_t packed_base = (size_t)line * {int(pitch) * int(support)};"
    return f"""{pad}int m_dense = world_constraint_count.data[world];
{pad}const int {rows} = m_dense;
{pad}const int dof_map_base = world * {dofs};
{pad}const int line = dense_first.data[world];
{pad}__builtin_assume(line >= 0);
{pad}const int off_dense = line * {pitch};
{pad}{base}"""


# ---------------------------------------------------------------------------------------------------------------------
# The Jacobi law: every row of a world at once, from the extrapolated impulses, with momentum
# ---------------------------------------------------------------------------------------------------------------------

JACOBI_UNILATERAL, JACOBI_FRICTION, JACOBI_BILATERAL, JACOBI_IDLE, JACOBI_FROZEN = 0, 1, 2, 3, -1
"""The kinds of a Jacobi row (``s_kind``): its projection, decoded from its type once at staging. A unilateral row
(contact normals, position limits, velocity limits) clamps at zero, a friction row's pair projects on its
disk and a bilateral row takes its step; a frozen row keeps its warm start. An idle row is a unilateral row at zero
impulse whose residual keeps it there: it cannot move, so it is out of the steps' bound until its residual turns
negative and it becomes unilateral again (:meth:`JacobiRows.idle`, :func:`jacobi_sweeps`)."""


@dataclasses.dataclass(frozen=True)
class JacobiRows:
    """A world's dense rows ``[0, rows)`` under the Jacobi law, a lane each (row ``i`` on lane ``i``, ``threads``
    lanes, at most ``capacity`` rows).

    Staged per row: its impulse ``s_x``; the extrapolated impulse the next step
    reads, ``s_y``; its right-hand side ``s_rhs``
    (once the operator has staged, the residual at zero impulse); its scaled
    step ``s_step``; its friction coefficient and parent; and its kind
    ``s_kind`` (:data:`JACOBI_UNILATERAL` and the others). ``s_x`` and ``s_y``
    are float4-aligned and padded with zeros to a multiple of four rows. With
    ``limits``, which rows are joint velocity limits are staged (``s_vlim``, for
    the last word: :func:`jacobi_limit_pass`). Every row is admitted
    provisionally: the operator forms its diagonal from the
    staged rows (walked in the kernel or read from the store), and a row whose formed diagonal is
    not positive is frozen once every lane has read the staged kinds (its
    lane's ``frozen``). The lane keeps its row's diagonal in a register
    (``row_diag``), which the exit law scales the row's impulse by; with
    ``types`` each row's constraint type is staged (``s_type``), which the
    trace names its change by. With ``weighted`` the lane also keeps whether
    its row is a contact (``row_contact``), whose weight the weighted Jacobi
    law reads (:func:`jacobi_sweeps`). Under ``torsion`` a contact's friction
    block is its pair and its spin row after it (contact torsion), else its
    pair.
    """

    capacity: int
    threads: int
    limits: bool = False
    rows: str = "n_rows"
    types: bool = False
    weighted: bool = False
    torsion: bool = False
    group: VirtualLanes = BLOCK_THREADS
    """The block's lanes: CUDA's threads, or the CPU's virtual lanes (:meth:`registers`)."""

    def registers(self) -> tuple[tuple[str, str], ...]:
        """Each lane's registers of its row: its diagonal, frozen flag and weight, its residual at zero impulse and
        step, and a sweep's update (:func:`jacobi_sweeps`)."""
        weighted = (("int", "row_contact"), ("float", "row_w")) if self.weighted else ()
        floats = (
            "row_diag",
            "row_rhs",
            "row_step",
            "row_inv_step",
            "row_share",
            "x_new",
            "x_old",
            "x_from",
            "row_bound",
        )
        return (
            *(("float", name) for name in floats),
            ("int", "frozen"),
            ("int", "row_joins"),
            *weighted,
        )

    @property
    def warp(self) -> bool:
        """Whether the rows' lanes are one warp's threads, which exchange values by shuffle."""
        return self.threads == 32 and not self.group.serial

    def buffers(self) -> tuple[Buffer, ...]:
        """The rows' state and metadata."""
        AM = self.capacity
        state = (Buffer("s_x", "float", AM, align=16), Buffer("s_y", "float", AM, align=16))
        floats = (Buffer(name, "float", AM) for name in ("s_rhs", "s_step", "s_mu"))
        ints = ("s_parent", "s_kind", *(("s_vlim",) if self.limits else ()), *(("s_type",) if self.types else ()))
        # Whether a row left idleness this sweep (:func:`jacobi_sweeps` sets the steps again after the sweep's vote).
        joined = Buffer("s_joined", "int", 1)
        return (*state, *floats, *(Buffer(name, "int", AM) for name in ints), joined)

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
        clamp at zero; under torsion a contact's spin row is of its friction pair's kind (stepped and projected with
        it)."""
        friction = f"row_type == {FRICTION} || row_type == {TORSION}" if self.torsion else f"row_type == {FRICTION}"
        decode = [
            f"if ({' || '.join(f'row_type == {kind}' for kind in unilateral)}) kind = {JACOBI_UNILATERAL};",
            f"else if ({friction}) kind = {JACOBI_FRICTION};",
        ]
        lines = [
            f"for (int i = lane; i < {self.rows}; i += {self.threads}) {{",
            "    const float l0 = world_impulses.data[off_dense + i];",
            "    s_x[i] = l0; s_y[i] = l0;",
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
            f"    row_contact = row_type == {CONTACT};" if self.weighted else "",
            "}",
        ]
        group = self.group
        lines.insert(0, group.declare("int", ("frozen", "0")))
        if self.weighted:
            lines.insert(0, group.declare("int", ("row_contact", "0")))
        # The lane's own row's diagonal (a row each: the loop runs at most once per lane).
        lines.insert(0, group.declare("float", ("row_diag", "0.0f")))
        return indent("\n".join(line for line in lines if line), pad)

    def freeze(self, pad: str = "    ") -> str:
        """Freeze the rows whose formed diagonal is not positive, after the staging's barrier; synchronized."""
        return self.group.phase(f"{pad}if (frozen) s_kind[lane] = {JACOBI_FROZEN};") + "\n" + self.group.barrier(pad)

    def idle(self, pad: str = "") -> str:
        """Mark each unilateral row at zero impulse with a non-negative residual at zero impulse idle, once the staging
        has formed ``s_rhs`` (each lane its own rows: no barrier). Its update clamps it at zero, so it moves nothing
        until its residual turns negative (:func:`jacobi_sweeps`)."""
        test = f"s_kind[i] == {JACOBI_UNILATERAL} && s_x[i] == 0.0f && s_rhs[i] >= 0.0f"
        return indent(
            f"for (int i = lane; i < {self.rows}; i += {self.threads})\n    if ({test}) s_kind[i] = {JACOBI_IDLE};\n"
            "if (lane == 0) s_joined[0] = 0;",
            pad,
        )

    @staticmethod
    def counted(row: str) -> str:
        """Whether row ``row`` counts in the steps' bound: admitted and not idle."""
        return f"(s_kind[{row}] >= 0 && s_kind[{row}] != {JACOBI_IDLE})"

    @staticmethod
    def clamps(row: str) -> str:
        """Whether row ``row`` clamps at zero: unilateral or idle."""
        return f"(s_kind[{row}] == {JACOBI_UNILATERAL} || s_kind[{row}] == {JACOBI_IDLE})"

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
    for a zero bound), ``bound`` a bound of the row's absolute sum ``sum_j |A_ij|`` over the rows that can move.

    ``w_i`` is the row's share of the response of the rows it couples to (generalized mass splitting), which keeps
    the simultaneous steps from diverging where many rows couple through one body: with ``S = diag(step)``,
    ``S^-1 - A`` is diagonally dominant on the moving rows, so ``S^-1 >= A`` there. An idle row moves nothing and is
    out of the sum (:meth:`JacobiRows.counted`); counted, the joint limits' rows of a tree, far from their limits,
    took most of its contacts' steps. The diagonal ``A_ii + cfm`` (the
    row's compliance) is formed here and a row without a positive one is marked for freezing (``frozen``,
    :meth:`JacobiRows.freeze`: the other lanes are still reading the kinds), but for a friction row: a tangent that
    moves nothing takes its block's step (:func:`isotropic_friction_step`) and still spends its share of the disk on
    its residual, as the sweeps' pair law does. The formed diagonal is the lane's ``row_diag``, which the exit law
    reads.
    """
    return indent(
        "\n".join(
            [
                f"const float w = ({bound} > 0.0f) ? fminf({aii} / {bound}, 1.0f) : 0.0f;",
                f"const float diag_i = {aii} + {cfm};",
                "row_diag = diag_i;",
                "if (diag_i > 0.0f) s_step[i] = w * omega / diag_i;",
                f"else {{ s_step[i] = 0.0f; frozen = s_kind[i] != {JACOBI_FRICTION}; }}",
            ]
        ),
        pad,
    )


def isotropic_friction_step(rows: JacobiRows, pad: str = "    ") -> str:
    """Give every row of a friction block one step in the block's scaled coordinates, then synchronize.

    A block is a contact's friction pair, and under torsion its spin row after it. Its friction law's set is the disk
    ``|t| <= mu lambda_n``, or the ball ``|(t / mu, s / mu_t)| <= lambda_n``: round in the coordinates
    ``(t, s mu / mu_t)``, where the radial projection (:func:`_jacobi_projection`) is the nearest point. Stepped
    isotropically there, by the smallest of the block's rows' steps in those coordinates, a block's fixed point
    ``x = P(x - S r)`` makes its friction oppose its slip in the law's metric: maximal dissipation, the sweeps' friction
    law. A row's step in its own coordinate is the block's step times its coordinate's scale squared: a tangent's 1, the
    spin row's ``(mu_t / mu)^2``. A row without a step of its own (a tangent that moves nothing) takes the block's; a
    block with none keeps its zero steps. Without a tangent coefficient (``mu = 0``) the projection pins the tangents
    and the spin row keeps its own step. The block's first tangent's lane handles it.
    """
    F = JACOBI_FRICTION
    lines = [
        f"for (int i = lane; i < {rows.rows}; i += {rows.threads}) {{",
        f"    if (s_kind[i] == {F} && i == s_parent[i] + 1) {{",
        "        const int j = i + 1;",
        f"        if (j < {rows.rows} && s_kind[j] == {F} && s_parent[j] == s_parent[i]) {{",
        "            const float step_a = s_step[i];",
        "            const float step_b = s_step[j];",
        # The block's step: the smallest positive one.
        "            float step_block = step_a > 0.0f ? step_a : step_b;",
        "            if (step_b > 0.0f) step_block = fminf(step_block, step_b);",
    ]
    if rows.torsion:
        lines += [
            "            const int k = i + 2;",
            "            const float mu = s_mu[i];",
            f"            if (k < {rows.rows} && s_kind[k] == {F} && s_parent[k] == s_parent[i] && mu > 0.0f) {{",
            # The spin coordinate's scale squared, (mu_t / mu)^2: zero pins the spin, whose step then stays.
            "                const float ratio = s_mu[k] / mu;",
            "                const float spin_scale = ratio * ratio;",
            "                const float step_spin = s_step[k];",
            "                if (spin_scale > 0.0f && step_spin > 0.0f) {",
            "                    const float spin_block = step_spin / spin_scale;",
            "                    step_block = step_block > 0.0f ? fminf(step_block, spin_block) : spin_block;",
            "                }",
            "                if (step_block > 0.0f && spin_scale > 0.0f) s_step[k] = step_block * spin_scale;",
            "            }",
        ]
    lines += [
        "            if (step_block > 0.0f) {",
        "                s_step[i] = step_block;",
        "                s_step[j] = step_block;",
        "            }",
        "        }",
        "    }",
        "}",
    ]
    return rows.group.phase(indent("\n".join(lines), pad)) + "\n" + rows.group.barrier(pad)


def _jacobi_join(rows: JacobiRows, op, scale, pad: str) -> str:
    """Rows that left idleness join the steps' bound, between two epochs of sweeps: an idle row whose residual turned
    negative kept its zero impulse in its sweep, voted loud and raised ``s_joined`` (:func:`_jacobi_step`), which ended
    the epoch after that sweep's vote; every admitted row's step is set again on the rows that can now move
    (:meth:`~.TransposedResponse.jacobi_steps`, then the friction blocks' isotropy), the extrapolation is dropped and the
    momentum restarts in the new metric, so the row moves from the next sweep under a bound that counts it: no row ever
    moves outside the bound. Out of the sweep loop, the join costs the sweeps nothing until it runs."""
    group = rows.group
    restep = [
        f"row_step = lane < {rows.rows} ? s_step[lane] : 0.0f;",
        "row_inv_step = row_step > 0.0f ? 1.0f / row_step : 0.0f;",
        "row_share = row_diag * row_step;",
        f"if (lane < {rows.rows}) s_y[lane] = s_x[lane];",
    ]
    lines = [
        group.barrier(""),
        op.jacobi_steps(scale, "", counted=rows.counted, first="if (lane == 0) s_joined[0] = 0;"),
        isotropic_friction_step(rows, ""),
        group.phase("\n".join(restep)),
        "t_k = 1.0f;",
        group.barrier(""),
    ]
    return indent("\n".join(lines), pad)


def _jacobi_step(rows: JacobiRows, op, laws: RowLaws, pad: str) -> str:
    """Row ``i``'s update ``x_new`` from the extrapolated impulses (``x_old``: its impulse), on its lane.

    Weighted laws step a row by its regularized step (:func:`.projection_laws.regularized_delta`) relaxed by its
    share ``s = omega w_i`` of a Gauss-Seidel step: ``y + s ((-r / d) w - (1 - w) y)``, with the delta and the impulse
    both scaled by ``s`` (the law is linear in them). For a material row of weight ``d / (d + eps)`` that is
    ``y - s (r + eps y) / (d + eps)``, whose fixed point is the row's ``r + eps lambda = 0`` whatever ``s``.
    """
    step = "value = s_y[i] - s_step[i] * r;"
    if laws.regularize:
        step = "value = s_y[i] + ProjectionLaw::regularized_delta(-row_step * r, row_w, row_share * s_y[i]);"
    update = [
        step,
        f"if {rows.clamps('i')} value = ProjectionLaw::unilateral(value);",
        # An idle row whose residual turned negative joins the bound after the sweep (_jacobi_join) and moves from the
        # next: it keeps its zero impulse now, its lane loud so that the world cannot stop before it has moved.
        f"if (s_kind[i] == {JACOBI_IDLE} && r < 0.0f) {{",
        f"    value = xi; row_joins = 1; s_kind[i] = {JACOBI_UNILATERAL}; s_joined[0] = 1;",
        "}",
    ]
    group = rows.group
    lines = [
        group.declare("float", *((name, "0.0f") for name in ("x_new", "x_old", "x_from", "row_bound"))),
        group.declare("int", ("row_joins", "0")),
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
    return group.phase(indent("\n".join(lines), pad))


def _jacobi_projection(rows: JacobiRows, *, pooled: bool, pad: str) -> str:
    """Project each friction row's update radially onto its pair's disk of radius ``mu`` times the normal load.

    The load is the pair's normal's fresh impulse (``pooled``: the sum over its patch ring of normals, in ring order
    from its own). One warp exchanges the fresh updates by shuffle (a pooled load reads the ring's staged ones, and
    the warp synchronizes before the impulses are stored again); several warps stage them in ``s_x``. The pair's
    tangents project together, each on its own lane. Under torsion the pair and its contact's spin row (its
    normal's row plus three) project together, radially onto their elliptic cone in the scaled impulse
    (:func:`.projection_laws.ball`).
    """
    if rows.torsion:
        return _jacobi_triple_projection(rows, pooled=pooled, pad=pad)
    F, n = JACOBI_FRICTION, rows.rows
    disk = "radius <= 0.0f ? 0.0f : ProjectionLaw::disk(wp::vec2({first}, {second}), radius)[0]"
    if rows.warp:
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
    project = [
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
    ]
    return _staged_projection(rows, project, pad)


def _staged_projection(rows: JacobiRows, project: list[str], pad: str) -> str:
    """Stage every row's fresh update in ``s_x``, then run each lane's ``project`` lines on the staged updates: the
    projection of a block of several warps, or of the CPU's virtual lanes."""
    group = rows.group
    lines = [
        group.barrier(""),
        group.phase(f"if (lane < {rows.rows}) s_x[lane] = x_new;"),
        group.barrier(""),
        group.phase("\n".join(project)),
        group.barrier(""),
    ]
    return indent("\n".join(lines), pad)


def _jacobi_triple_projection(rows: JacobiRows, *, pooled: bool, pad: str) -> str:
    """:func:`_jacobi_projection` where spin rows may appear: each friction-kind lane (a tangent or the spin row of
    normal ``p``) projects the triple ``(p + 1, p + 2, p + 3)`` onto its cone and keeps its own component; a contact
    without a spin row (:func:`spin_row`: no friction-kind row of the same normal at ``p + 3``) projects its pair as the
    triple of a zero spin coefficient, its pair's disk."""
    F, n = JACOBI_FRICTION, rows.rows
    # The Jacobi kinds give a spin row its pair's: a friction-kind row of the same normal after the pair is one.
    spun = spin_row("p", n, lambda row: f"(s_kind[{row}] == {F} && s_parent[{row}] == p ? 1 : 0)", spin="1") + " >= 0"
    project = [
        "        const float radius_load = fmaxf({load}, 0.0f);",
        "        const wp::vec3 triple = ConeLaw::ball(wp::vec3({t1}, {t2}, spun ? {s} : 0.0f), s_mu[p + 1],",
        "                                              spun ? s_mu[p + 3] : 0.0f, radius_load);",
        "        x_new = triple[i - p - 1];",
    ]
    if rows.warp:
        lines = [
            "{",
            "    const int i = lane;",
            f"    const bool fr = (i < {n}) && (s_kind[i] == {F});",
            "    const int p = fr ? s_parent[i] : i;",
            f"    const bool spun = fr && {spun};",
            "    const float x1 = READ(x_new, (fr ? p + 1 : i) & 31);",
            "    const float x2 = READ(x_new, (fr ? p + 2 : i) & 31);",
            "    const float x3 = READ(x_new, (spun ? p + 3 : i) & 31);",
        ]
        load = "xp"
        if pooled:
            lines += [f"    if (i < {n}) s_x[i] = x_new;", "    WARP_SYNC();", "    if (fr) {"]
            lines.append(indent(f"const float load = {rows.ring_load('p')};", "        "))
            load = "load"
        else:
            lines += ["    const float xp = READ(x_new, p & 31);", "    if (fr) {"]
        lines += [line.format(load=load, t1="x1", t2="x2", s="x3") for line in project]
        lines += ["    }", "    WARP_SYNC();" if pooled else "", "}"]
        return indent("\n".join(line for line in lines if line), pad)
    ring = [indent(f"const float load = {rows.ring_load('p')};", "        ")] if pooled else []
    lines = [
        "{",
        "    const int i = lane;",
        f"    if (i < {n} && s_kind[i] == {F}) {{",
        "        const int p = s_parent[i];",
        f"        const bool spun = {spun};",
        *ring,
        *(
            line.format(load="load" if pooled else "s_x[p]", t1="s_x[p + 1]", t2="s_x[p + 2]", s="s_x[p + 3]")
            for line in project
        ),
        "    }",
        "}",
    ]
    return _staged_projection(rows, lines, pad)


def _jacobi_momentum(rows: JacobiRows, *, pad: str) -> str:
    """Nesterov's extrapolation factor ``beta`` with adaptive restart: the momentum restarts (``t_k = 1``) when the
    last step points against it, ``(y - x_new) . (x_new - x_old) > 0`` summed over the rows (per warp, then over the
    warps in order).

    The sequence's next term and factor depend on ``t_k`` alone, so they are formed ahead of the reduction, off its
    chain, and the restart selects its own (``beta = 0`` and the term after ``t = 1``, which the compiler folds)."""
    lines = [
        "const float t_next = 0.5f * (1.0f + sqrtf(1.0f + 4.0f * t_k * t_k));",
        "const float beta_next = (t_k - 1.0f) / t_next;",
    ]
    term = "(s_y[lane] - x_new) * (x_new - x_old)"
    if rows.group.serial:
        lines.append(rows.group.sum("dot", f"lane < {rows.rows} ? {term} : 0.0f", ""))
    else:
        lines += [
            "float dot_local = 0.0f;",
            f"if (lane < {rows.rows}) dot_local = {term};",
            "float dot = dot_local;",
            "SUM(dot);",
        ]
    block = rows.threads > 32 and not rows.group.serial
    if block:
        lines += [
            "if ((lane & 31) == 0) s_dot[lane >> 5] = dot;",
            "SYNC();",
            "dot = 0.0f;",
            "#pragma unroll",
            f"for (int wv = 0; wv < {rows.threads // 32}; ++wv) dot += s_dot[wv];",
        ]
    lines += [
        "const bool restart = dot > 0.0f;",
        "const float beta = restart ? 0.0f : beta_next;",
        "t_k = restart ? 0.5f * (1.0f + sqrtf(5.0f)) : t_next;",
    ]
    if block:
        lines.append("SYNC();")
    return indent("\n".join(lines), pad)


def jacobi_sweeps(rows: JacobiRows, op, laws: RowLaws, *, sweeps: int, pooled: bool, scale, pad: str = "    ") -> str:
    """The Jacobi sweeps of a world's rows over the response operator ``op`` (its Jacobi actions), at most ``sweeps``.

    The rows of each friction block first take one step, isotropic in the block's scaled coordinates
    (:func:`isotropic_friction_step`). Each sweep, every admitted row steps at once from the extrapolated impulses
    ``y``: its residual ``r = b' + (A y)_i`` (:meth:`~.TransposedResponse.jacobi_residual`, after the operator's
    :meth:`~.TransposedResponse.gather`); then ``y_i - s_step_i r`` clamped at zero for a unilateral or idle row (an idle
    row whose residual turned negative stays at zero, votes loud and joins the steps' bound after the sweep's vote,
    :func:`_jacobi_join`, the steps set again by ``scale`` as the staging set them). Each friction block is then
    projected radially onto its disk, or under torsion its ball (``pooled``: on its patch ring's pooled normal load), the
    impulses advance with Nesterov's
    momentum and adaptive restart, each row votes its change under the exit law (:func:`change_vote`:
    its step from the extrapolated impulse over its step size, which is the row's residual there for an unclamped row,
    against its floor; the trace names it by the row's staged type, :attr:`JacobiRows.types`), and the sweep ends
    under the policy (:func:`end_sweep`, the budget ``sweeps``; a block of several warps reduces the trace with
    ``BLOCK_ARGMAX``, :func:`.solve_policy.block_argmax`).
    """
    policy = laws.policy
    if policy.traces and not rows.types:
        raise ValueError("the trace names the Jacobi rows' changes by their staged types")
    if laws.regularize and not rows.weighted:
        raise ValueError("the weighted Jacobi law reads the weights of the contact rows its store marks")
    if laws.torsion != rows.torsion:
        raise ValueError("the Jacobi rows' friction blocks follow the laws' torsion")
    group = rows.group
    weight = ""
    if laws.regularize:
        # The lane's row's weight, fixed through the sweeps: a contact's stored one, or a material contact's from the
        # diagonal the tier formed (row_weight); 1 for every other row.
        stored = "world_row_w.data[off_dense + lane]"
        if laws.material:
            stored = row_weight(stored, "world_row_eps.data[off_dense + lane]", "row_diag")
        weight = f"{pad}{group.declare('const float', ('row_w', f'row_contact ? {stored} : 1.0f'))}"
    inner = pad + "    "
    hot = inner + "    "
    change = Change("x_new - x_from", "row_bound", "s_type[i]" if rows.types else "-1", "row_inv_step")
    # A block of several warps votes and reduces the trace through the block; the CPU's one thread holds them all.
    block = rows.threads > 32 and not group.serial
    budget = str(int(sweeps))
    # A sweep: every row's step from the extrapolated impulses, the blocks' projection and the momentum, then each
    # row's store and vote.
    vote = [
        f"{hot}{{",
        f"{hot}    const int i = lane;",
        f"{hot}    if (i < {rows.rows}) {{",
        f"{hot}        s_x[i] = x_new;",
        f"{hot}        s_y[i] = x_new + beta * (x_new - x_old);",
        f"{hot}        if (row_joins) iteration_changed = 1;",
        f"{hot}        if (x_new != x_from) {{",
        change_vote(laws, (change,), hot + "            "),
        f"{hot}        }}",
        f"{hot}    }}",
        f"{hot}}}",
    ]
    vote = [group.phase("\n".join(vote)), group.barrier(hot)]
    phases = (
        op.gather(hot),
        _jacobi_step(rows, op, laws, hot),
        _jacobi_projection(rows, pooled=pooled, pad=hot),
        _jacobi_momentum(rows, pad=hot),
        "\n".join(vote),
    )
    # The exit law; a row that joined the steps' bound ends its epoch after its sweep's vote (the flag it raised is
    # read alike by every lane), and the join sets the next epoch's steps.
    exit_ = (
        end_sweep(
            policy, iteration="sweep", block=block, iterations=budget, argmax="BLOCK_ARGMAX" if block else "ARGMAX"
        ).rstrip("\n")
        + f"\n{hot}if (s_joined[0]) {{ ++sweep; break; }}"
    )
    # The lane's row: its residual at zero impulse, fixed through the sweeps, and its step size, set again when a row
    # joins the steps' bound; the step's share of a Gauss-Seidel step (omega w_i): the bound's impulse term in the
    # step's units.
    row = [
        f"{pad}{group.declare('const float', ('row_rhs', f'lane < {rows.rows} ? s_rhs[lane] : 0.0f'))}",
        f"{pad}{group.declare('float', ('row_step', f'lane < {rows.rows} ? s_step[lane] : 0.0f'))}",
        f"{pad}{group.declare('float', ('row_inv_step', 'row_step > 0.0f ? 1.0f / row_step : 0.0f'))}",
        f"{pad}{group.declare('float', ('row_share', 'row_diag * row_step'))}",
        weight,
    ]
    lines = [
        rows.freeze(pad),
        isotropic_friction_step(rows, pad),
        group.phase("\n".join(line for line in row if line)),
        f"{pad}float t_k = 1.0f;",
        # The sweeps run in epochs of one metric (level_loop's resumed counter).
        f"{pad}int sweep = 0;",
        f"{pad}for (;;) {{",
        level_loop(phases, exit=exit_, iteration="sweep", budget=budget, resume=True, pad=inner),
        f"{inner}if (sweep >= {sweeps} || !s_joined[0]) break;",
        _jacobi_join(rows, op, scale, inner),
        f"{pad}}}",
    ]
    return "\n".join(line for line in lines if line)


def jacobi_limit_pass(rows: JacobiRows, op, *, pad: str = "    ") -> str:
    """The last word of the joint velocity-limit rows after the final Jacobi sweep, on the first warp, in row order.

    With ``r_i = b'_i + (A x)_i``, the residual the sweeps use, a violated admitted row (``r_i < 0``) is pushed onto
    its limit, ``x_i += -r_i / A_ii`` (the row law's update from a zero impulse,
    :func:`.projection_laws.velocity_limit_update`, added to the row's total). The pass only pushes: it changes
    nothing at the sweeps' fixed point, where no row is violated, and gives the limits the last word of a world
    stopped early, as the Gauss-Seidel sweeps' end pass does. The staged rows' walked support class forms
    ``(A x)_i = Z_i . w`` on ``w = Y^T x`` (:meth:`~.TransposedResponse.push_prologue`) with ``A_ii = |Z_i|^2 + cfm``
    in one ladder, and each push adds its change to ``w`` (:meth:`~.TransposedResponse.staged`).
    """
    if not rows.limits:
        raise ValueError("the last word pushes the velocity-limit rows the rows stage")
    # The first warp pushes; on the CPU the one thread is the group's every lane.
    group = rows.group
    warp = 1 if group.serial else 32
    pushed = op.staged(Lanes(warp))
    body = "            "
    lines = [
        "{",
        group.barrier("    "),
        op.push_prologue("    "),
        f"    if (lane < {warp}) {{",
        f"        for (int i = 0; i < {rows.rows}; ++i) {{",
        f"            if (!{rows.clamps('i')} || s_vlim[i] == 0) continue;",
        pushed.dot((("part", "row", None), ("zz", "row", "row")), {"row": "i"}, body),
        "            const float r_vlim = s_rhs[i] + part;",
        "            const float diag_vlim = zz + cfm;",
        "            const float delta_vlim = diag_vlim > 0.0f",
        "                ? ProjectionLaw::velocity_limit_update(0.0f, r_vlim / diag_vlim).v1 : 0.0f;",
        "            if (delta_vlim != 0.0f) {",
        "                if (lane == 0) s_x[i] += delta_vlim;",
        pushed.axpy((("row", "delta_vlim"),), body + "    ", row="i"),
        "            }",
        "            WARP_SYNC();",
        "        }",
        "    }",
        group.barrier("    "),
        "}",
    ]
    return indent("\n".join(line for line in lines if line), pad)


# ---------------------------------------------------------------------------------------------------------------------
# The configurations and the one factory
# ---------------------------------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, kw_only=True)
class SweepConfig:
    """A sweep kernel's configuration: the switches a class's sweep is built from, frozen, so equal configurations
    are one kernel (:func:`sweep_kernel`).

    Every configuration carries the row laws in force (:attr:`laws`: the solve policy, weighted rows, material
    contacts and spin rows, from the options and the model, never from a route); each class's configuration adds
    its own switches. A configuration declares what the factory reads of it: its native ABI (:attr:`ABI`, its
    first parameters the dispatch's), its native function's name (:attr:`NATIVE`), the kernel's name rendered from
    its fields (:attr:`name`), its body and working set (:meth:`source`), the wrapper's dispatch (:meth:`dispatch`),
    its operands beyond the ABI (:meth:`operands`) and its launch (:meth:`work`).

    Args:
        solve_policy: The solve policy: the readout beside the exit law.
        weighted: Contact rows step by their regularization weights.
        material: Material contacts step by their compliance's weights (they are weighted).
        torsion: Spin rows may follow friction pairs.
    """

    solve_policy: SolvePolicy = DEFAULT_POLICY
    weighted: bool = False
    material: bool = False
    torsion: bool = False

    ABI: ClassVar[tuple]
    """The native ABI (:func:`.kernel_template.params`)."""
    NATIVE: ClassVar[str]
    """The native function's name."""
    LABEL: ClassVar[str]
    """What the working set's budget check calls the sweep."""

    @property
    def laws(self) -> RowLaws:
        """The row laws the configuration compiles."""
        return RowLaws(regularize=self.weighted, material=self.material, torsion=self.torsion, policy=self.solve_policy)

    @property
    def name(self) -> str:
        """The kernel's name, rendered from the configuration's fields."""
        raise NotImplementedError

    @property
    def label(self) -> str:
        """What the working set's budget check calls the sweep."""
        return self.LABEL

    def native(
        self,
        body: str,
        *,
        lanes: int,
        group: int | None = None,
        ring: bool = True,
        rows: tuple[DevicePrelude, ...] = (),
        walks: tuple[DevicePrelude, ...] = (),
    ) -> str:
        """The sweep's native function: ``body`` after the frame's preludes, composed for a world of ``lanes`` lanes
        (the CPU's one thread runs them in turn, :func:`.device_preludes.group_native`).

        The preludes, in order: the lane group of ``group`` lanes (by default ``lanes``), the shared stage's storage,
        the patch ring's walks (``ring``), the rows' own preludes (``rows``), the row laws (the projections, and the
        cone's where spin rows may appear), the solve policy's, then the walks the sweep compiles (``walks``).
        """
        laws = law_preludes(PROJECTION_LAW, *((CONE_LAW,) if self.torsion else ()))
        preludes = (
            lane_group(lanes if group is None else group),
            SHARED_STORAGE,
            *((RING,) if ring else ()),
            *rows,
            *laws,
            *self.solve_policy.preludes,
            *walks,
        )
        return group_native(body, preludes=preludes, lanes=lanes)

    def law_operands(self) -> tuple[RowArgument, ...]:
        """The operands the row laws add beyond the ABI: the rows' weights where the laws are weighted and the ABI
        holds none, material contacts' compliance, and the solve policy's readout."""
        holds_weights = any(name == "world_row_w" for name, _ in self.ABI)
        return (
            *weight_arguments(self.weighted and not holds_weights),
            *material_arguments(self.material),
            *self.solve_policy.arguments,
        )

    def source(self) -> tuple[str, Staging]:
        """The native body and its working set."""
        raise NotImplementedError

    def dispatch(self, staging: Staging) -> Dispatch:
        """How the kernel's threads reach their worlds (:func:`.kernel_template.items`)."""
        raise NotImplementedError

    def operands(self) -> tuple[RowArgument, ...]:
        """The operands beyond the ABI (:func:`.kernel_template.operands`)."""
        return ()

    def work(self, items: int = 0, shared: int = 0) -> Work:
        """What the kernel's launches over ``items`` worlds declare (:class:`~newton._src.core.execution.Work`), its
        block's working set ``shared``: its launch bound is the layer's shape of it at no items."""
        raise NotImplementedError


@functools.cache
def sweep_kernel(config: SweepConfig) -> wp.Kernel:
    """The one sweep factory: the kernel of ``config`` (:class:`SweepConfig`).

    The configuration's body and working set, the working set checked against the static budget of the device it is
    built for (:func:`.memory.arch_budget`), the launch bound the layer's shape of its work there
    (:func:`~newton._src.core.execution.shape`), the kernel its native ABI under its dispatch.
    """
    snippet, staging = config.source()
    staging.check(config.label, arch_budget(config.device_arch))
    bounds = shape(config.work(0, staging.bytes), built_for(config.device_arch)).bounds
    return KernelTemplate.from_abi(
        config.name,
        config.ABI,
        snippet,
        config.dispatch(staging),
        extra=config.operands(),
        composed=True,
        native_name=config.NATIVE,
        launch_bounds=bounds,
    ).kernel()

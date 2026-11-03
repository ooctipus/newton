# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Author response actions independently of force production and publication.

Vector operands are functions from an index expression to the native expression
of the element at that index. The caller binds storage offsets in them. The rows
are whitened by the walk (:mod:`.walk_rows`), and the velocity change in factor
coordinates leaves by the walk (:func:`.tree_factor.walk_leave_kernel`).

A tree's factor is leaf-first: ``L`` of ``P H P^T`` with ``P`` the reversal of an
articulation's coordinates (factor coordinate ``i`` is physical coordinate
``n - 1 - i``). A vector in factor coordinates that shares a physical vector's
storage sits in physical positions, its factor coordinate ``i`` at position
``n - 1 - i`` (:func:`leaf_first`).

These generators allocate no storage and impose no launch boundary. Owners
supply scratch, lane ownership, synchronization between dependent actions,
held-factor validity and output transactions. Operands are authored bindings,
never extracted from another owner's source. No runtime method is selected here.
"""

from collections.abc import Callable
from dataclasses import dataclass

from .solve_policy import Change, bound, terms_bound
from .staging import Buffer

Vector = Callable[[str], str]
"""A vector operand: the native expression of its element at an index expression."""

Matrix = Callable[[str, str], str]
"""A matrix operand: the native expression of its element at a (row, column) pair of index expressions."""


def _at(vector: Vector, index: str) -> str:
    return vector(f"({index})")


def leaf_first(size: int | str, index: str) -> str:
    """The position of factor coordinate ``index`` in an ``size``-coordinate vector: ``size - 1 - index`` (the
    leaf-first factor's coordinate reversal)."""
    return f"({size}-1-({index}))"


def _operand(expression: str) -> str:
    """``expression`` as an operand: parenthesized unless it is a name or a literal."""
    return expression if expression.replace("_", "x").replace(".", "x").isalnum() else f"({expression})"


def indent_lines(text: str, pad: str) -> str:
    """``text`` with every non-empty line indented by ``pad``."""
    return "\n".join(pad + line if line.strip() else line for line in text.split("\n"))


@dataclass(frozen=True)
class Lanes:
    """The threads that update one row together.

    ``width`` lanes of the sweep's lane group (``LANE`` in ``lane``) split a
    row's sums and reduce them with ``SUM``; every lane then holds the result
    and runs the row's scalar law redundantly. ``width == 1`` is one thread
    per row: sums run in term order and nothing is reduced or synchronized.
    With ``segment``, the lanes are an aligned segment of ``width`` lanes of a
    wider group, one row per segment, and sums reduce over the segment
    (``SEGMENT_SUM``).
    """

    width: int = 32
    lane: str = "lane"
    segment: bool = False

    @property
    def parallel(self) -> bool:
        return self.width > 1

    def reduce(self, *values: str) -> str:
        """Statements that reduce ``values`` over the lanes (nothing for one thread)."""
        if not self.parallel:
            return ""
        if self.segment:
            return f"SEGMENT_SUM({self.width}, {', '.join(values)});"
        return f"SUM({', '.join(values)});"

    def sync(self) -> str:
        """A barrier of the row's lanes (nothing for one thread)."""
        return "SYNC();" if self.parallel else ""

    def first(self, statement: str) -> str:
        """``statement`` run by the row's first lane only."""
        return f"if ({self.lane} == 0) {statement}" if self.parallel else statement

    def broadcast(self, value: str) -> str:
        """``value`` as the first lane reads it, on every lane."""
        return f"READ({self.lane} == 0 ? {value} : 0.0f, 0)" if self.parallel else value


@dataclass(frozen=True)
class Velocity:
    """Where a sweep's rows read and update the velocity they act on: ``array[index]``."""

    array: str = "s_v"

    def at(self, index: str) -> str:
        return f"{self.array}[{index}]"


class ResponseOperator:
    """A row response representation: residual, diagonal, coupling and apply, emitted per row.

    A concrete operator reads a row's diagonal through :attr:`diagonal`
    (``A_ii``) and, when its builder stores or stages it, :attr:`inverse`
    (``1 / A_ii``): callables from a row index to the native expression. The
    step multiplies by the inverse when there is one and divides by the
    diagonal otherwise, as the builders of the sweeps it replaces left them.

    The row actions are native statements at the indentation ``indent``; an
    operator implements the ones its sweeps take. The visited row is ``i``.
    """

    name = "response"
    residual_includes_rhs = False
    """Whether :meth:`residual` gives the whole residual ``J v + rhs`` (a maintained residual) or ``J v``."""
    applies = True
    """Whether :meth:`apply` writes state the row's lanes share (the velocity, a maintained residual), so the lanes
    synchronize after it. An impulse-space operator without apply leaves every lane its own copy of the impulses."""
    fused_pair_apply = False
    """Whether a row's apply takes its friction pair's second tangent too (``apply_pair``: the sibling's change, then
    the row's, per entry, written back once) instead of ``apply_row`` of the sibling then ``apply`` of the row."""
    diagonal: Callable[[str], str] | None = None
    inverse: Callable[[str], str] | None = None

    def value(self, row: str) -> str:
        """``A_ii`` of row ``row`` (zero for a zero inverse)."""
        if self.diagonal is not None:
            return self.diagonal(row)
        inverse = self.inverse(row)
        return f"{inverse} > 0.0f ? 1.0f / {inverse} : 0.0f"

    def bound(self, rhs: str, terms: str, impulse: str, row: str) -> str:
        """The exit law's bound of row ``row`` at ``impulse`` (:func:`.solve_policy.bound`), in the form its storage
        gives without a division: impulse units from a stored inverse, velocity units from a stored diagonal."""
        if self.inverse is not None:
            return bound(rhs, terms, impulse, step=self.inverse(row))
        return bound(rhs, terms, impulse, diagonal=self.diagonal(row))

    def change(self, delta: str, bound: str, kind: str, row: str) -> Change:
        """Row ``row``'s change ``delta`` against its :meth:`bound`, as the exit law votes it."""
        return Change(delta, bound, kind, self.value(row), scaled=self.inverse is None)

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

    def velocity_term(self, residual: str, rhs: str, jv: str) -> str:
        """``J v`` of the row: the reduced ``jv``, or the maintained residual less its right-hand side."""
        return f"{residual} - {rhs}" if self.residual_includes_rhs else jv

    def terms(self, out: str, rhs: str, *, whole: bool) -> str:
        """``S_i`` of a sum this operator reduced into ``out``, the magnitude of the terms its velocity sum adds, or a
        lower bound of it, which the exit law scales a row's floor by (:class:`.solve_policy.Change`).

        By default the reduced velocity term: ``|out|``, or ``|out - rhs|`` for a sum that includes its right-hand
        side ``rhs`` (``whole``). An operator that reduces ``sum_k |t_ik|`` beside the sum returns it.
        """
        return terms_bound(out, rhs) if whole else f"fabsf({out})"

    # -- row actions ---------------------------------------------------------------------------------------------------

    def _lacks(self, action: str):
        return NotImplementedError(f"{type(self).__name__} has no {action}")

    def residual(self, out: str, indent: str) -> str:
        """``float out = J_i v`` of the visited row (every lane holds it), or its whole maintained residual
        (:attr:`residual_includes_rhs`)."""
        raise self._lacks("residual")

    def row_velocity(self, out: str, row: str, indent: str) -> str:
        """``float out = J_row v`` of a row outside the visit (every lane holds it)."""
        raise self._lacks("row velocity")

    def sibling_residual(self, out: str, sibling: str, indent: str, rhs: str) -> str:
        """``float out = J_s v + rhs`` of the friction sibling ``sibling``: its row velocity, then its right-hand
        side ``rhs``. An operator that maintains residuals reads the sibling's whole."""
        return f"{self.row_velocity(out, sibling, indent)}\n{indent}{out} += {rhs};"

    def coupling(self, out: str, row: str, sibling: str, indent: str) -> str:
        """``float out = J_i Y_s``: the pair's off-diagonal entry."""
        raise self._lacks("pair coupling")

    def pair_entries(self, sibling: str, indent: str) -> str:
        """Declare what a friction pair's terms and its second tangent's apply read of the sibling ``sibling``, once,
        before the pair's disk: nothing by default (:meth:`pair_terms` and :meth:`apply_row` read their own)."""
        return ""

    def pair_terms(self, row: str, sibling: str, rhs: Callable[[str], str], indent: str, *, own: bool = False) -> str:
        """Declare a friction pair's ``sibling_residual`` (``J_s v + rhs_s``) and ``cross`` (``J_i Y_s``).

        With ``own`` the row's residual is reduced here too: ``residual`` (``J_i v + rhs_i``) of the operator's
        current row, for a pair solved as one unit. ``rhs`` maps a row index to its right-hand side. An operator
        whose lanes reduce several sums in one ladder reduces the pair's sums together.
        """
        lines = []
        if own:
            lines.append(self.residual("residual", indent))
            lines.append("" if self.residual_includes_rhs else f"{indent}residual += {rhs(row)};")
        lines += [
            self.sibling_residual("sibling_residual", sibling, indent, rhs(sibling)),
            self.coupling("cross", row, sibling, indent),
        ]
        return "\n".join(line for line in lines if line != "")

    def spin_terms(self, row: str, sibling: str, spin: str, rhs: Callable[[str], str], indent: str) -> str:
        """Declare a friction triple's spin terms (contact torsion), after :meth:`pair_terms`: ``spin_residual``
        (``J_p v + rhs_p`` of the spin row ``spin``) and its couplings with the tangents, ``spin_cross``
        (``J_i Y_p``) and ``spin_cross_sibling`` (``J_s Y_p``)."""
        lines = (
            self.sibling_residual("spin_residual", spin, indent, rhs(spin)),
            self.coupling("spin_cross", row, spin, indent),
            self.coupling("spin_cross_sibling", sibling, spin, indent),
        )
        return "\n".join(line for line in lines if line != "")

    def apply(self, delta: str, indent: str) -> str:
        """``v += Y_i delta`` of the visited row."""
        raise self._lacks("apply")

    def apply_row(self, row: str, delta: str, indent: str) -> str:
        """``v += Y_row delta`` of a row outside the visit (the friction sibling)."""
        raise self._lacks("row apply")

    def apply_pair(self, sibling_delta: str, delta: str, indent: str) -> str:
        """``v += Y_s sibling_delta + Y_i delta`` in one pass (:attr:`fused_pair_apply`)."""
        raise self._lacks("pair apply")

    def apply_strided(self, row: str, delta: str, indent: str) -> str:
        """``v += Y_row delta`` after :meth:`row_velocity` of the same row (the end-of-sweep passes): by default
        :meth:`apply_row`; an operator may reuse what the row velocity kept."""
        return self.apply_row(row, delta, indent)

    # -- Jacobi actions: every row of a block at once, one lane a row (row ``i = lane``) -------------------------------
    #
    # A Jacobi sweep (:func:`.sweep.jacobi_sweeps`) keeps each row's residual at zero impulse, ``b'_i = rhs_i + J_i
    # (v - Y^T lambda_0)``, in ``s_rhs`` and forms ``r_i(y) = b'_i + (A y)_i`` at the extrapolated impulses ``y``
    # (``s_y``) each sweep. The velocity is published once, from the impulses the sweeps applied; the last word of
    # the velocity-limit rows pushes them on ``A x``, maintained as ``w = Y^T x`` where the operator keeps one.

    def registers(self, indent: str) -> str:
        """Declare the lanes' row registers (nothing by default)."""
        return ""

    def jacobi_stage(
        self,
        padding: str,
        scale: Callable[[str, str, str], str],
        indent: str,
        *,
        idle: str,
        counted: Callable[[str], str],
    ) -> str:
        """Stage the operator, set each row's ``s_rhs`` to ``b'_i``, run ``idle`` (statements that read ``s_rhs``:
        the row store's marks of the rows that cannot move) and set each admitted row's step (:meth:`jacobi_steps`);
        ``padding`` zeroes the rows' padded impulses ahead of the last barrier."""
        raise self._lacks("Jacobi staging")

    def jacobi_steps(self, scale: Callable[[str, str, str], str], indent: str, *, counted: Callable[[str], str]) -> str:
        """Set each admitted row's step, ``scale(aii, bound, indent)``: from ``A_ii`` and a bound of its absolute sum
        ``sum_j |A_ij|`` over the rows ``counted(j)`` names (those that can move); synchronized. Run at staging and
        whenever a row joins the counted rows."""
        raise self._lacks("Jacobi steps")

    def gather(self, indent: str) -> str:
        """Per sweep, before the rows update: the products every row's residual reads (nothing by default)."""
        return ""

    def jacobi_residual(self, out: str, indent: str) -> str:
        """``float out = b'_i + (A y)_i`` of row ``i = lane``."""
        raise self._lacks("Jacobi residual")

    def push_prologue(self, indent: str) -> str:
        """Before the last word's pushes: ``w = Y^T x`` where the operator keeps it (nothing by default)."""
        return ""

    def push_velocity(self, out: str, lanes: Lanes, indent: str) -> str:
        """``float out = (A x)_i`` of the pushed row ``i`` over ``lanes``: the residual's impulse term."""
        raise self._lacks("push velocity")

    def push_apply(self, delta: str, lanes: Lanes, indent: str) -> str:
        """After a push ``delta`` of row ``i``: ``w += Y_i delta`` where the operator keeps ``w`` (nothing by
        default)."""
        return ""

    def push_diagonal(self) -> str:
        """``A_ii`` of the pushed row ``i``."""
        raise self._lacks("push diagonal")


@dataclass(frozen=True)
class FreeBodyZ(ResponseOperator):
    """Rows over at most two free bodies, whitened: ``Z_a, Z_b = L^-1 J^T`` of 6 coordinates each.

    A free body is a one-node tree; ``L`` is its held factor (``H = L L^T``,
    armature included), and the sweep holds the change of the body's six
    velocity coordinates in factor coordinates, ``u = L^T dv``, until
    :meth:`leave`: a row's velocity term is ``Z^T u``, its update ``u += Z
    delta``. Row ``i``'s entries are at ``[base + i * 6 + k]``; its bodies'
    first velocity coordinates are ``dof_a``/``dof_b`` (negative for no body).
    With lanes, lanes 0-5 own body ``a``'s coordinates and lanes 6-11 body
    ``b``'s, one product each, and the lanes reduce with ``SUM``; the other
    lanes add zero. With one thread (``Lanes(1)``) the visited row ``i``'s
    entries are read where they are used: the residual adds each body's six
    products as one expression, ``jv += Z_0 u_0 + ... + Z_5 u_5``, body
    ``a``'s then ``b``'s; a friction pair's
    sibling residual starts from its right-hand side and its coupling from
    zero, each adding body ``a``'s then ``b``'s product per coordinate, as does
    the row's own residual where the pair reduces it (``own``). The builder
    stores ``1 / A_ii`` (``A_ii = cfm + |Z_a|^2 + |Z_b|^2``).

    ``factors`` names the native :class:`.kernels.BodyFactors` operand; a
    body's speed row reads its velocity by its walk (``walk_leave``,
    ``walk_enter``).
    """

    lanes: Lanes = Lanes()
    Z_a: str = "mf_Z_a"
    Z_b: str = "mf_Z_b"
    base: str = "mf6_base"
    velocity: Velocity = Velocity()
    diagonal: Callable[[str], str] | None = None
    inverse: Callable[[str], str] | None = None
    entry: Callable[[str, str, str], str] | None = None
    """One thread: where entry ``k`` of a row's ``Z_a`` or ``Z_b`` (the array's name) is read, ``entry(array, row,
    k)``, when a mapping stages the rows elsewhere (default: the array, by row)."""
    factors: str = "factors"
    walk_leave: Callable[[str, str], str] | None = None
    """The code mapping the six values ``values`` of the body whose first coordinate is ``first`` back in place,
    ``walk_leave(first, values)``, ``v = P^T L^-T u`` by the walk down (:mod:`.walk_rows`); a speed row needs it."""
    walk_enter: Callable[[str, str, str], str] | None = None
    """With ``walk_leave``: the code giving ``u = L^T P v`` of the six values ``velocity`` in ``out``,
    ``walk_enter(first, velocity, out)``."""
    name = "free_body_z"

    def __post_init__(self):
        if self.entry is not None and self.lanes.parallel:
            raise ValueError("staged entries take the one-thread layout")

    def pack(self, word: str, dof_a: str, dof_b: str, inverse: str, rhs: str, kind: str, parent: str, indent: str):
        """Declare ``int4 word``, a row's packed word as :meth:`consume` decodes it.

        ``x``: the bodies' first world coordinates, ``dof_a`` in the high and ``dof_b`` in the low 16 bits (-1 for
        none); ``y`` and ``z``: the bits of the stored inverse diagonal and of the right-hand side; ``w``: the row
        type in the low and the parent (0xFFFF for none) in the high 16 bits.
        """
        return (
            f"{indent}int4 {word};\n"
            f"{indent}{word}.x = ({dof_a} << 16) | ({dof_b} & 0xFFFF);\n"
            f"{indent}{word}.y = __float_as_int({inverse});\n"
            f"{indent}{word}.z = __float_as_int({rhs});\n"
            f"{indent}{word}.w = static_cast<int>(static_cast<unsigned int>({kind}) | "
            f"(static_cast<unsigned int>({parent}) << 16));"
        )

    # -- software pipeline: the packed row word (bodies, inverse diagonal, rhs, type and parent) and the entries ---

    def pipeline_registers(self, indent: str = "        ") -> str:
        return f"{indent}int4 pre_meta;\n{indent}float pre_Za = 0.0f, pre_Zb = 0.0f;"

    def _load(self, row_meta: str, row6: str, indent: str) -> str:
        lane = self.lanes.lane
        return (
            f"{indent}pre_meta = *reinterpret_cast<const int4*>(&mf_meta.data[{row_meta}]);\n"
            f"{indent}if ({lane} < 6) pre_Za = {self.Z_a}.data[{row6} + {lane}];\n"
            f"{indent}if ({lane} >= 6 && {lane} < 12) pre_Zb = {self.Z_b}.data[{row6} + {lane} - 6];"
        )

    def prefetch_first(self, first: str, indent: str) -> str:
        """Load row ``first``'s packed word and entries into the pipeline registers."""
        if first == "0":
            return self._load("off_meta", self.base, indent)
        return self._load(f"off_meta + {first} * 4", f"{self.base} + {first} * 6", indent)

    def consume(self, row: str, end: str, indent: str) -> str:
        """Take row ``row``'s word and entries from the pipeline, load row ``row + 1``'s, and decode the word.

        Declares ``meta``, ``cur_*``, ``dof_a``, ``dof_b``, ``mf_diag`` (the stored inverse diagonal),
        ``packed_tp`` and ``mf_rt`` (the row type).
        """
        return (
            f"{indent}int4 meta = pre_meta;\n"
            f"{indent}float cur_Za = pre_Za;\n{indent}float cur_Zb = pre_Zb;\n"
            f"{indent}if ({row} + 1 < {end}) {{\n"
            f"{indent}    int next_mf6 = {self.base} + ({row} + 1) * 6;\n"
            + self._load(f"off_meta + ({row} + 1) * 4", "next_mf6", indent + "    ")
            + f"\n{indent}}}\n"
            f"{indent}int packed_dofs = meta.x;\n"
            f"{indent}int dof_a = packed_dofs >> 16;\n"
            f"{indent}int dof_b = (packed_dofs << 16) >> 16;\n"
            f"{indent}float mf_diag = __int_as_float(meta.y);\n"
            f"{indent}int packed_tp = meta.w;\n"
            f"{indent}int mf_rt = packed_tp & 0xFFFF;"
        )

    def _lane_terms(self, out: str, a_value: str, b_value: str, indent: str, assign: str = "=") -> str:
        lane = self.lanes.lane
        return (
            f"{indent}if ({lane} < 6 && dof_a >= 0)\n{indent}    {out} {assign} {a_value};\n"
            f"{indent}if ({lane} >= 6 && {lane} < 12 && dof_b >= 0)\n{indent}    {out} {assign} {b_value};"
        )

    def residual(self, out: str, indent: str) -> str:
        """``float out = Z_i u`` from the pipelined entries ``cur_Za``/``cur_Zb`` (one thread: row ``i``'s)."""
        lane = self.lanes.lane
        if self.lanes.parallel:
            return (
                f"{indent}float {out}_partial = 0.0f;\n"
                + self._lane_terms(
                    f"{out}_partial",
                    f"cur_Za * {self.velocity.at(f'dof_a + {lane}')}",
                    f"cur_Zb * {self.velocity.at(f'dof_b + {lane} - 6')}",
                    indent,
                )
                + f"\n{indent}{self.lanes.reduce(f'{out}_partial')}\n{indent}float {out} = {out}_partial;"
            )
        sides = "\n".join(
            f"{indent}if ({body} >= 0) {{\n{indent}    {out} += "
            + " + ".join(f"{self._at(Z, 'i', k)} * {self._v(body, k)}" for k in range(6))
            + f";\n{indent}}}"
            for Z, body in ((self.Z_a, "dof_a"), (self.Z_b, "dof_b"))
        )
        return f"{indent}float {out} = 0.0f;\n{sides}"

    def apply(self, delta: str, indent: str) -> str:
        """``u += Z_i delta`` from the pipelined entries ``cur_Za``/``cur_Zb`` (one thread: row ``i``'s)."""
        if not self.lanes.parallel:
            return self.apply_row("i", delta, indent)
        lane = self.lanes.lane
        return (
            f"\n{indent}if ({lane} < 6 && dof_a >= 0) {{\n"
            f"{indent}    {self.velocity.at(f'dof_a + {lane}')} += cur_Za * {delta};\n{indent}}}\n"
            f"{indent}if ({lane} >= 6 && {lane} < 12 && dof_b >= 0) {{\n"
            f"{indent}    {self.velocity.at(f'dof_b + {lane} - 6')} += cur_Zb * {delta};\n{indent}}}"
        )

    def row_velocity(self, out: str, row: str, indent: str) -> str:
        """``float out = Z_row u`` of a row outside the pipeline (lanes 0-11, reduced; one thread: from zero)."""
        if not self.lanes.parallel:
            return self._per_coordinate(out, "0.0f", self._velocity_term(row), indent)
        lane = self.lanes.lane
        entries = f"{self.base} + {row} * 6"
        return (
            f"{indent}float {out} = 0.0f;\n"
            + self._lane_terms(
                out,
                f"{self.Z_a}.data[{entries} + {lane}] * {self.velocity.at(f'dof_a + {lane}')}",
                f"{self.Z_b}.data[{entries} + {lane} - 6] * {self.velocity.at(f'dof_b + {lane} - 6')}",
                indent,
            )
            + f"\n{indent}{self.lanes.reduce(out)}"
        )

    def sibling_residual(self, out: str, sibling: str, indent: str, rhs: str) -> str:
        """``float out = Z_s u + rhs``: the row velocity, then the right-hand side; one thread sums from it."""
        if not self.lanes.parallel:
            return self._per_coordinate(out, rhs, self._velocity_term(sibling), indent)
        return super().sibling_residual(out, sibling, indent, rhs)

    def coupling(self, out: str, row: str, sibling: str, indent: str) -> str:
        """``float out = Z_i Z_s`` (lanes 0-11, reduced; one thread: from zero)."""
        if not self.lanes.parallel:
            return self._per_coordinate(
                out, "0.0f", lambda Z, body: f"{self._at(Z, row, 'k')} * {self._at(Z, sibling, 'k')}", indent
            )
        lane = self.lanes.lane
        i_row, s_row = f"{self.base} + {row} * 6", f"{self.base} + {sibling} * 6"
        return (
            f"{indent}float {out} = 0.0f;\n"
            + self._lane_terms(
                out,
                f"{self.Z_a}.data[{i_row} + {lane}] * {self.Z_a}.data[{s_row} + {lane}]",
                f"{self.Z_b}.data[{i_row} + {lane} - 6] * {self.Z_b}.data[{s_row} + {lane} - 6]",
                indent,
            )
            + f"\n{indent}{self.lanes.reduce(out)}"
        )

    def pair_terms(self, row: str, sibling: str, rhs: Callable[[str], str], indent: str, *, own: bool = False) -> str:
        """The pair's terms; one thread reduces the row's own residual (``own``) as the sibling's: per coordinate,
        body ``a``'s then ``b``'s product, from its right-hand side."""
        if not own or self.lanes.parallel:
            return super().pair_terms(row, sibling, rhs, indent, own=own)
        return "\n".join(
            (
                self._per_coordinate("residual", rhs(row), self._velocity_term(row), indent),
                self.sibling_residual("sibling_residual", sibling, indent, rhs(sibling)),
                self.coupling("cross", row, sibling, indent),
            )
        )

    def apply_row(self, row: str, delta: str, indent: str) -> str:
        """``u += Z_row delta`` for a row outside the pipeline (lanes 0-11; one thread: each coordinate in order)."""
        if not self.lanes.parallel:
            return "".join(
                f"\n{indent}if ({body} >= 0) {{"
                + "".join(f"\n{indent}    {self._v(body, k)} += {self._at(Z, row, k)} * {delta};" for k in range(6))
                + f"\n{indent}}}"
                for Z, body in ((self.Z_a, "dof_a"), (self.Z_b, "dof_b"))
            )
        lane = self.lanes.lane
        s_row = f"{self.base} + {row} * 6"
        return (
            f"\n{indent}if ({lane} < 6 && dof_a >= 0)\n"
            f"{indent}    {self.velocity.at(f'dof_a + {lane}')} += {self.Z_a}.data[{s_row} + {lane}] * {delta};\n"
            f"{indent}if ({lane} >= 6 && {lane} < 12 && dof_b >= 0)\n"
            f"{indent}    {self.velocity.at(f'dof_b + {lane} - 6')} += {self.Z_b}.data[{s_row} + {lane} - 6] * {delta};"
        )

    # -- one thread per row: entries read where they are used -------------------------------------------------------

    def _at(self, array: str, row: str, k) -> str:
        if self.entry is not None:
            return self.entry(array, row, str(k))
        return f"{array}.data[{self.base} + {row} * 6 + {k}]"

    def _v(self, first: str, k) -> str:
        return self.velocity.at(f"{first} + {k}" if k != 0 else first)

    def _velocity_term(self, row: str):
        return lambda Z, body: f"{self._at(Z, row, 'k')} * {self._v(body, 'k')}"

    def _per_coordinate(self, out: str, start: str, term, indent: str) -> str:
        """``float out = start``, then per coordinate ``k`` the term ``term(Z, body)`` of body ``a``, then of body
        ``b`` (each side when it has a body)."""
        sides = "\n".join(
            f"{indent}    if ({body} >= 0) {out} += {term(Z, body)};"
            for Z, body in ((self.Z_a, "dof_a"), (self.Z_b, "dof_b"))
        )
        return f"{indent}float {out} = {start};\n{indent}for (int k = 0; k < 6; ++k) {{\n{sides}\n{indent}}}"

    # -- free-body speed rows ---------------------------------------------------------------------------------------

    def speed(self, out: str, body: str, axis: str, indent: str) -> str:
        """``const float out``: the magnitude of the body's physical velocity coordinates ``[axis, axis + 3)``.

        The body's coordinates from ``body`` hold ``u``; every lane forms its physical velocity ``speed_v = L^-T u``
        (``walk_leave``) on the predictor, which ``v_out`` holds through the sweep, and the magnitude, the squares added
        in coordinate order. The lanes then synchronize, so :meth:`scale_speed` may write ``u``.
        """
        if self.walk_leave is None:
            raise ValueError("a free body's speed row reads its velocity by its walk (walk_leave)")
        lines = [
            "float speed_v[6];",
            "for (int k = 0; k < 6; ++k) speed_v[k] = " + self._v(body, "k") + ";",
            self.walk_leave(body, "speed_v"),
            f"for (int k = 0; k < 6; ++k) speed_v[k] += v_out.data[world_dof_indices.data[dof_map_base + {body} + k]];",
            f"const float {out} = sqrtf((speed_v[{axis}] * speed_v[{axis}] + speed_v[{axis} + 1] * speed_v[{axis} + 1])"
            f" + speed_v[{axis} + 2] * speed_v[{axis} + 2]);",
        ]
        if self.lanes.parallel:
            lines.append(self.lanes.sync())
        return "\n".join(indent + line for line in "\n".join(lines).split("\n"))

    def scale_speed(self, scale: str, body: str, axis: str, indent: str) -> str:
        """``v += scale v`` on the coordinates :meth:`speed` read, as ``u += L^T dv`` (one thread, or the first lane)."""
        lines = [
            "float speed_dv[6];",
            f"for (int k = 0; k < 6; ++k) speed_dv[k] = k >= {axis} && k < {axis} + 3 ? {scale} * speed_v[k] : 0.0f;",
        ]
        if self.walk_enter is None:
            raise ValueError("a free body's speed row pushes its velocity by its walk (walk_enter)")
        lines += [
            "float speed_du[6];",
            self.walk_enter(body, "speed_dv", "speed_du"),
            "for (int k = 0; k < 6; ++k) " + self._v(body, "k") + " += speed_du[k];",
        ]
        text = "\n".join(lines)
        if self.lanes.parallel:
            text = f"if ({self.lanes.lane} == 0) {{\n{text}\n}}"
        return "\n".join(indent + line for line in text.split("\n"))


class ResidualBlock(ResponseOperator):
    """The arithmetic of a staged response block ``A`` (``s_A``) whose rows keep their whole residual (``s_res``).

    Row ``r``'s residual ``rhs + J_r v`` is maintained in ``s_res[r]``: a
    change ``delta`` of row ``i`` adds ``A[i][r] delta`` to every row ``r`` of
    the block (``rows``), lane-strided, and the pair coupling ``J_i Y_s`` is the
    entry ``A[s][i]``. A layout places entry ``A[i][r]`` (:meth:`entry`); a view
    of a later range of the block's rows maps its row indices to block rows
    (:meth:`block_row`).
    """

    residual_includes_rhs = True

    def block_row(self, row: str) -> str:
        """The block row of the view's row ``row``."""
        return row

    def entry(self, source: str, target: str) -> tuple[str, str]:
        """``(statements, index)`` of ``A[source][target]``, the response of row ``target`` to row ``source``."""
        raise NotImplementedError

    def residual(self, out: str, indent: str) -> str:
        return f"{indent}float {out} = s_res[{self.block_row('i')}];"

    def sibling_residual(self, out: str, sibling: str, indent: str, rhs: str) -> str:
        """The sibling's maintained residual, which holds its right-hand side ``rhs`` already."""
        return f"{indent}float {out} = s_res[{self.block_row(sibling)}];"

    def coupling(self, out: str, row: str, sibling: str, indent: str) -> str:
        """``J_i Y_s``, the pair's off-diagonal entry, as the block stores it (``A[s][i]``)."""
        setup, index = self.entry(self.block_row(sibling), self.block_row(row))
        lead = f"{indent}{setup}\n" if setup else ""
        return f"{lead}{indent}float {out} = s_A[{index}];"

    def spin_terms(self, row: str, sibling: str, spin: str, rhs: Callable[[str], str], indent: str) -> str:
        """The spin row's maintained residual and its couplings with the tangents (``A[p][i]``, ``A[p][s]``), each
        entry read in a scope of its own: a layout's setup declares its indices."""
        lines = [self.sibling_residual("spin_residual", spin, indent, rhs(spin))]
        for out, source in (("spin_cross", row), ("spin_cross_sibling", sibling)):
            setup, index = self.entry(self.block_row(spin), self.block_row(source))
            read = f"{out} = s_A[{index}];"
            lines.append(f"{indent}float {out};\n{indent}{{ {setup} {read} }}" if setup else f"{indent}float {read}")
        return "\n".join(lines)

    def apply_row(self, row: str, delta: str, indent: str) -> str:
        lane, width = self.lanes.lane, self.lanes.width
        setup, index = self.entry(self.block_row(row), "r")
        head = f"{indent}for (int r = {lane}; r < {self.rows}; r += {width})"
        update = f"s_res[r] += s_A[{index}] * {delta};"
        if not setup:
            return f"{head} {update}"
        return f"{head} {{\n{indent}    {setup}\n{indent}    {update}\n{indent}}}"

    def apply(self, delta: str, indent: str) -> str:
        return self.apply_row("i", delta, indent)


def _applied_sum(out: str, start: str, response: Callable[[str], str], applied: Callable[[str], str], rows: str):
    """Lines of ``float out = start + sum_i response(i) applied(i)`` over rows ``[0, rows)`` in row order, a zero
    applied impulse skipped."""
    return [
        f"float {out} = {start};",
        f"for (int i = 0; i < {rows}; ++i) {{",
        f"    const float a = {applied('i')};",
        f"    if (a != 0.0f) {out} += {response('i')} * a;",
        "}",
    ]


def _lines(lines, indent: str) -> str:
    return "\n".join(indent + line if line.strip() else line for line in "\n".join(lines).split("\n"))


# The last word of a Jacobi sweep's velocity-limit rows (:func:`.sweep.jacobi_limit_pass`) on ``w``, the velocity
# change of the current impulses: ``Y`` (or ``Z``) at ``(row, coordinate)`` through ``response``, the pushed row's ``J``
# at a coordinate through ``row``.


def _push_prologue(response: Callable[[str, str], str], lanes: Lanes, dofs: int, rows: str):
    """Lines of ``w = Y^T x`` (``s_dv``), lane-strided over the coordinates, each a sum over the rows in order."""
    return [
        f"for (int d = {lanes.lane}; d < {dofs}; d += {lanes.width}) {{",
        "    float acc = 0.0f;",
        f"    for (int j = 0; j < {rows}; ++j) acc += {response('j', 'd')} * s_x[j];",
        "    s_dv[d] = acc;",
        "}",
        "SYNC();",
    ]


def _push_velocity(out: str, row: Callable[[str], str], lanes: Lanes, dofs: int):
    """Lines of ``float out = Z_i w`` and ``zz = |Z_i|^2`` over ``lanes``, in one ladder."""
    return [
        f"float {out} = 0.0f, zz = 0.0f;",
        f"for (int d = {lanes.lane}; d < {dofs}; d += {lanes.width}) {{",
        f"    const float row = {row('d')};",
        f"    {out} += row * s_dv[d];",
        "    zz += row * row;",
        "}",
        lanes.reduce(out, "zz"),
    ]


def _push_apply(delta: str, column: Callable[[str], str], lanes: Lanes, dofs: int):
    """Lines of ``w += Y_i delta`` over ``lanes``."""
    return [f"for (int d = {lanes.lane}; d < {dofs}; d += {lanes.width}) s_dv[d] += {column('d')} * {delta};"]


@dataclass(frozen=True)
class WalkedRows:
    """A world's rows in the walk's coordinates, built in the kernel: each row's seeds walked up its trees' root paths
    (:func:`.tree_factor.walk_paths_prelude`), ``Z_i`` at the row's world coordinates.

    A world holds the ``primary`` tree of ``primary`` coordinates from world coordinate ``primary_offset`` and at most
    one ``secondary`` tree from ``secondary_offset``: its articulations ``art_a`` and ``art_b`` (``ink_meta``, ``-1``
    for none). Each lane takes its row of ``rows``: the row input's statements ``read`` write its footprints at
    ``Zi[q * ZS]`` (world coordinate ``q``, a staged row's coordinates ``stride`` apart) and its incident velocity
    ``J_i v_hat`` into ``incident``, which folds into the row's right-hand side. A row input that walks its rows in the
    kernel calls ``TIER_WALK(side, body, k, seed)``: the wrench ``seed[0:6]`` at local ``body`` of the world's tree
    ``side`` (``k < 0``), or the covector ``seed[0] e_k`` at its local coordinate ``k`` (``body`` its body), walked into
    the row and its ``J`` entries against ``v_hat`` into ``incident``; with ``walked`` each slipping tree's release ``R``
    (``walk.release``, :class:`.tree_factor.WalkRelease`) then takes the decided factor's row, as the rows the one
    writer stored hold it already. ``frame`` is each tree's slip capacity, 0 where none of the class's trees of that
    size can slip: its release compiles out. ``A = Z^T Z``, so ``Z`` serves as both ``J^T`` and ``Y``; the lane keeps
    ``Z_i`` in its registers (``Jr``). The velocity change of the applied impulses, ``u = Z^T delta``, leaves by the
    walk down in the rows' square root, ``P^T L^-T R^T u``, a tree per lane (:meth:`publish`).
    """

    primary: int
    secondary: int
    primary_offset: int
    secondary_offset: int
    dofs: int
    stride: int
    lanes: Lanes
    read: str
    walked: bool = False
    frame: tuple[int, int] = (0, 0)
    rows: str = "n_rows"

    def _tree(self) -> list[str]:
        """The world's trees (every lane): ``art_a``, ``art_b``."""
        return ["const int art_a = ink_meta.data[world * 4];", "const int art_b = ink_meta.data[world * 4 + 2];"]

    def _walk(self) -> str:
        """``TIER_WALK``: one seed of row ``i`` on the world's tree ``side`` (:func:`.tree_factor.walk_into`'s walk), its
        footprints added at the tree's world coordinates and its ``J`` entries against ``s_v`` into ``incident``."""
        from .tree_factor import WALK_COORDINATE, WALK_LINK, WALK_NOW  # noqa: PLC0415

        return f"""const auto TIER_WALK = [&](const int side, const int body, const int k, const float* seed) {{
    const int art = side == 0 ? art_a : art_b, offset = side == 0 ? {self.primary_offset} : {self.secondary_offset};
    const int j0 = walk.joint_start.data[art], d0 = walk.dof_start.data[art];
    WALK_TREE(reinterpret_cast<const int*>(walk.path.data + j0), walk.joint_end.data[art] - j0, walk.dofs.data[art]);
    const float* data = walk.data.data + (size_t)d0 * {WALK_COORDINATE};
    const float* link = walk.link.data + (size_t)j0 * {WALK_LINK};
    // The step's geometry the finish published: its columns and links at the step's poses.
    const float* now = walk.now.data + (size_t)d0 * {WALK_NOW};
    const float* now_link = walk.now_link.data + (size_t)j0 * {WALK_LINK};
    const auto sink = [&](int, int c, float z, float j) {{
        Zi[(offset + c) * ZS] += z;
        incident += j * s_v[offset + c];
    }};
    float shift[3];
    WALK_SHIFT(walk, art, shift);
    WALK_ROW(data, link, now, now_link, shift, body, k, seed, sink);
}};"""

    def _sides(self) -> tuple[tuple[int, int, str], ...]:
        """The trees whose slips' release compiles: ``(side, coordinates, articulation)``."""
        trees = ((0, self.primary, self.frame[0], "art_a"), (1, self.secondary, self.frame[1], "art_b"))
        return tuple((side, n, art) for side, n, cap, art in trees if n and cap)

    def _release_rows(self) -> list[str]:
        """The release on the lane's row, each side whose tree slips: ``z <- R z`` (``WALK_RELEASE`` on the row's
        coordinates, ``ZS`` apart), the decided factor's row, inside the row's support."""
        lines = []
        for side, _, art in self._sides():
            offset = self.primary_offset if side == 0 else self.secondary_offset
            lines.append(f"if ({art} >= 0) WALK_RELEASE(walk.release, {art}, &Zi[{offset} * ZS], ZS);")
        return lines

    def stage(self, target: str, *, indent: str) -> str:
        """Walk each lane's row into ``target`` (``Zi = &target[i]``) and its registers, its incident velocity folded
        into its right-hand side."""
        D = self.dofs
        row = [
            f"if ({self.lanes.lane} < {self.rows}) {{",
            *("    " + line for line in self._tree()),
            f"    const int i = {self.lanes.lane};",
            f"    float* Zi = &{target}[i];",
            f"    constexpr int ZS = {self.stride};",
            "    #pragma unroll",
            f"    for (int d = 0; d < {D}; ++d) Zi[d * ZS] = 0.0f;",
            "    // J_i v_hat, folded into the row's right-hand side.",
            "    float incident = 0.0f;",
        ]
        if self.walked:
            row.append(_lines([self._walk()], "    "))
        row += ["    {", self.read, "    }", "    s_rhs[i] += incident;"]
        if self.walked:
            row.append(_lines(self._release_rows(), "    "))
        row.append("}")
        # Each lane's registers take its row only now (a lane past the rows zero), so they are not live during the walk.
        row += [
            "#pragma unroll",
            f"for (int d = 0; d < {D}; ++d) Jr[d] = {self.lanes.lane} < {self.rows} ? {target}[d * {self.stride} + {self.lanes.lane}] : 0.0f;",
        ]
        return _lines(row, indent)

    def publish(self, z: Callable[[str], str], applied: Callable[[str], str], indent: str) -> str:
        """``u = sum_i Z_i applied_i`` (``z(i)``: ``Z_i`` at coordinate ``d``), in row order, in the walk's coordinates
        to the step's one leave (:attr:`.tree_factor.WalkTrees.du`, :func:`.tree_factor.walk_leave_kernel`), which
        walks each tree down: ``v = v_in + P^T L^-T R^T u``."""
        lane, width, D = self.lanes.lane, self.lanes.width, self.dofs
        lines = [
            f"for (int d = {lane}; d < {D}; d += {width}) {{",
            *("    " + line for line in _applied_sum("u", "0.0f", z, applied, self.rows)),
            "    const int global_dof = world_dof_indices.data[dof_map_base + d];",
            "    if (global_dof >= 0) walk.du.data[global_dof] += u;",
            "}",
        ]
        return _lines(lines, indent)

    def push_velocity(self, out: str, z: Callable[[str, str], str], lanes: Lanes, indent: str) -> str:
        """``float out = Z_i w`` and ``zz = |Z_i|^2`` of the pushed row ``i`` over ``lanes``, in one ladder."""
        return _lines(_push_velocity(out, lambda d: z("i", d), lanes, self.dofs), indent)

    def push_diagonal(self) -> str:
        """``A_ii = |Z_i|^2 + cfm_i``, from the squares :meth:`push_velocity` reduced."""
        return "zz + world_row_cfm.data[off_dense + i]"


@dataclass(frozen=True)
class TransposedResponse(ResponseOperator):
    """Walked rows ``Z`` in registers and staged transposed: ``A y = Z^T (Z y)``, one lane a row.

    The rows are walked in the kernel or read from the one writer's store (:class:`WalkedRows`), so ``Z``
    serves as both ``J^T`` and ``Y``. Each lane holds its row in registers
    (``Jr``); the block stages ``Z`` coordinate-major, ``s_Yt[d * stride + i]
    = Z_i[d]`` (``stride = capacity + 4``: past the rows by a float4, so that
    neighbouring coordinates fall in different banks), with zero columns up to
    a multiple of four rows. Each sweep forms ``w = Z y`` once
    (:meth:`gather`): a lane per coordinate and chunk of the rows' float4
    groups (``chunks`` of them, as many as the lanes cover), four accumulators
    joined pairwise, the chunks summed in order; a row's residual is then
    ``b'_i + Z_i w`` in coordinate order. The step scale's bound is the
    column-sum bound ``sum_d |Z_id| sum_j |Z_jd|`` over the rows that can move
    (:meth:`jacobi_steps`), the bound the tiers' sweep budget was fit with
    (:data:`.sweep_routes_tiers.TIER_SWEEPS_PER_ITERATION`).
    """

    dofs: int
    capacity: int
    walked: WalkedRows
    lanes: Lanes = Lanes()
    rows: str = "n_rows"

    def __post_init__(self):
        if self.walked is None:
            raise ValueError("the transposed response holds walked rows")

    diagonal: Callable[[str], str] | None = None
    inverse: Callable[[str], str] | None = None
    name = "transposed_response"

    @property
    def stride(self) -> int:
        return self.capacity + 4

    @property
    def chunks(self) -> int:
        """Row chunks of :meth:`gather`: lanes per coordinate."""
        return max(self.lanes.width // self.dofs, 1)

    def _yt(self, row: str, coordinate: str) -> str:
        return f"s_Yt[{coordinate} * {self.stride} + {row}]"

    def buffers(self, capacity: int) -> tuple[Buffer, ...]:
        """The staged block, the column sums of the bound, ``w`` and the chunks' partial ``w``."""
        D = self.dofs
        buffers = [Buffer("s_Yt", "float", self.stride * D, align=16), Buffer("s_colsum", "float", D)]
        buffers.append(Buffer("s_dv", "float", (D + 3) & ~3, align=16))
        if self.chunks > 1:
            buffers.append(Buffer("s_dvp", "float", self.chunks * D))
        return tuple(buffers)

    def registers(self, indent: str) -> str:
        """Declare each lane's row registers ``Jr``, which the stage fills (:meth:`WalkedRows.stage`)."""
        return f"{indent}float Jr[{self.dofs}];"

    def jacobi_stage(
        self,
        padding: str,
        scale: Callable[[str, str, str], str],
        indent: str,
        *,
        idle: str,
        counted: Callable[[str], str],
    ) -> str:
        """Walk and stage the rows, mark the rows that cannot move (``idle``) and set each step scale
        (:meth:`jacobi_steps`): ``s_rhs[i]``, which took ``J_i v_hat`` at staging, is the row's residual ``b'_i`` at
        zero impulse; ``padding`` pads the rows' impulses.
        """
        lane, width, D, AM, rows = self.lanes.lane, self.lanes.width, self.dofs, self.capacity, self.rows
        stage = [self.walked.stage("s_Yt", indent=indent)]
        lines = [
            f"for (int e = {lane}; e < {D} * 4; e += {width}) {{",
            "    const int d = e >> 2;",
            f"    const int j = {rows} + (e & 3);",
            f"    if (j < {AM}) {self._yt('j', 'd')} = 0.0f;",
            "}",
            idle,
            "SYNC();",
            *self._jacobi_steps(scale, counted),
            padding,
            "SYNC();",
        ]
        return "\n".join([*stage, _lines(lines, indent)])

    def jacobi_steps(self, scale: Callable[[str, str, str], str], indent: str, *, counted: Callable[[str], str]) -> str:
        """Set each admitted row's step from the column-sum bound over the counted rows; synchronized."""
        return _lines([*self._jacobi_steps(scale, counted), "SYNC();"], indent)

    def _jacobi_steps(self, scale: Callable[[str, str, str], str], counted: Callable[[str], str]) -> list[str]:
        """The column sums over the counted rows, then each admitted row's ``A_ii``, bound and step (lane ``i``'s row
        in its registers)."""
        lane, width, D, rows = self.lanes.lane, self.lanes.width, self.dofs, self.rows
        return [
            f"for (int d = {lane}; d < {D}; d += {width}) {{",
            "    float csum = 0.0f;",
            f"    for (int j = 0; j < {rows}; ++j)",
            f"        if ({counted('j')}) csum += fabsf({self._yt('j', 'd')});",
            "    s_colsum[d] = csum;",
            "}",
            "SYNC();",
            f"for (int i = {lane}; i < {rows}; i += {width}) {{",
            "    const float* Ji = Jr;",
            "    if (s_kind[i] >= 0) {",
            "        float aii = 0.0f, bound = 0.0f;",
            "        #pragma unroll",
            f"        for (int d = 0; d < {D}; ++d) {{",
            f"            aii += Ji[d] * {self._yt('i', 'd')};",
            "            bound += fabsf(Ji[d]) * s_colsum[d];",
            "        }",
            scale("aii", "bound", "        "),
            "    }",
            "}",
        ]

    def gather(self, indent: str) -> str:
        """``w = Y^T y`` (``s_dv``): a lane per coordinate and chunk of float4 row groups, four accumulators joined
        pairwise, the chunks' partial sums (``s_dvp``) added in chunk order; synchronized."""
        lane, width, D, chunks = self.lanes.lane, self.lanes.width, self.dofs, self.chunks
        store = "s_dv[d]" if chunks == 1 else f"s_dvp[ch * {D} + d]"
        lines = [
            "{",
            f"    const int n4 = ({self.rows} + 3) >> 2;",
            f"    const int per = (n4 + {chunks - 1}) / {chunks};",
            f"    if ({lane} < {chunks * D}) {{",
            f"        const int d = {lane} % {D};",
            f"        const int ch = {lane} / {D};",
            "        const float4* y4 = reinterpret_cast<const float4*>(s_y);",
            f"        const float4* Y4 = reinterpret_cast<const float4*>(&s_Yt[d * {self.stride}]);",
            "        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;",
            "        const int j4_end = min(n4, (ch + 1) * per);",
            "        for (int j4 = ch * per; j4 < j4_end; ++j4) {",
            "            const float4 yv = y4[j4];",
            "            const float4 Yv = Y4[j4];",
            "            a0 += Yv.x * yv.x; a1 += Yv.y * yv.y; a2 += Yv.z * yv.z; a3 += Yv.w * yv.w;",
            "        }",
            f"        {store} = (a0 + a1) + (a2 + a3);",
            "    }",
            "    SYNC();",
        ]
        if chunks > 1:
            lines += [
                f"    for (int d = {lane}; d < {D}; d += {width}) {{",
                "        float acc = 0.0f;",
                "        #pragma unroll",
                f"        for (int ch = 0; ch < {chunks}; ++ch) acc += s_dvp[ch * {D} + d];",
                "        s_dv[d] = acc;",
                "    }",
                "    SYNC();",
            ]
        lines.append("}")
        return _lines(lines, indent)

    def jacobi_residual(self, out: str, indent: str) -> str:
        """``float out = b'_i + J_i w``, in coordinate order from ``b'_i``."""
        lines = [
            "const float* Ji = Jr;",
            f"float {out} = s_rhs[i];",
            "#pragma unroll",
            f"for (int d = 0; d < {self.dofs}; ++d) {out} += Ji[d] * s_dv[d];",
        ]
        return _lines(lines, indent)

    def publish(self, indent: str = "    ", *, applied: Callable[[str], str]) -> str:
        """``v = v_hat + P^T L^-T sum_i Z_i applied_i`` in row order from the staged block."""
        return self.walked.publish(lambda row: self._yt(row, "d"), applied, indent)

    def push_prologue(self, indent: str) -> str:
        """``w = Y^T x`` before the pushes of the last word."""
        return _lines(_push_prologue(self._yt, self.lanes, self.dofs, self.rows), indent)

    def push_velocity(self, out: str, lanes: Lanes, indent: str) -> str:
        """``float out = Z_i w`` and ``zz = |Z_i|^2`` over ``lanes``."""
        return self.walked.push_velocity(out, self._yt, lanes, indent)

    def push_apply(self, delta: str, lanes: Lanes, indent: str) -> str:
        """``w += Y_i delta`` after a push of row ``i``."""
        return _lines(_push_apply(delta, lambda d: self._yt("i", d), lanes, self.dofs), indent)

    def push_diagonal(self) -> str:
        """``A_ii`` of a pushed row, formed from its row."""
        return self.walked.push_diagonal()


@dataclass(frozen=True)
class SparsePairZ(ResponseOperator):
    """Whitened rows over a world's dense group of ``dofs`` coordinates plus two scalar sparse coordinates.

    The sparse-diagonal rows: a world pairs one compact dense articulation (its group ``group``, first coordinate
    ``offset``) with diagonal-mass coordinates, and the sweep holds the velocity change in factor coordinates: ``u =
    R^-T L^T P dv`` on the dense articulation (the walk's factor and its slips' release) and ``u = sqrt(m) dv`` on a
    diagonal coordinate (a star's factor is diagonal). Row ``i``'s dense part is its walk's footprints ``Z = F L^-1 P
    J^T`` at ``Z[(group * rows + i) * dofs + l]`` on coordinate ``offset + l`` (the walk's coordinate ``l``); its sparse
    part is two slots ``s``: coordinate ``sparse_dof[(base + i) * 2 + s]`` (``-1`` for none, below ``-1`` an
    independent chain's link) and ``Z_s = J_s / sqrt(m_s)``, a key's one step, at ``sparse_z[(base + i) * 2 + s]``. A row's velocity term is ``Z^T u``, its update
    ``u += Z delta``, and two rows couple through ``Z_i^T Z_s``: ``W = Z^T Z``.

    With lanes (a warp, or a segment of ``width`` lanes), lane ``l < dofs`` owns dense coordinate ``l`` and lanes
    ``dofs`` and ``dofs + 1`` the slots: each lane's one product, zero without a coordinate, is reduced with the
    lanes' sum; the other lanes add zero. A segment of at least ``dofs + 2`` lanes reduces the same products in the
    same order as the warp, up to the sign of a zero sum. The visited row's lane entries are ``coord`` and ``z``
    (:meth:`consume`).

    With one thread (``Lanes(1)``), a row on the single coordinate ``coordinate`` (an independent scalar contact,
    whose dense part is empty), through the slot that holds it: its product is rounded on its own (``__fmul_rn``,
    no contraction into the right-hand side's add), as the lanes' sum of one product rounds it.

    Written with the rows (:func:`.walk_rows.write_slot_rows`, :func:`.compact_contact.produce_contacts`), ``A_ii =
    |Z|^2 + cfm`` is stored as a value; the step divides by it.
    """

    dofs: int
    rows: int
    lanes: Lanes = Lanes()
    Z: str = "dense_Z"
    group: str = "dense_group"
    offset: str = "dense_offset"
    sparse_dof: str = "sparse_row_dof"
    sparse_z: str = "sparse_row_z"
    base: str = "row_base"
    coordinate: str = "scalar_coord"
    velocity: Velocity = Velocity()
    diagonal: Callable[[str], str] | None = None
    inverse: Callable[[str], str] | None = None
    name = "sparse_pair_z"

    def _entries(self, row: str, coord: str, value: str, indent: str, active=None) -> str:
        """Declare this lane's coordinate and entry of row ``row`` (``coord`` negative for none)."""
        lane, P = self.lanes.lane, self.dofs
        guard = f"{active} && " if active else ""
        lines = [
            f"int {coord} = -1;",
            f"float {value} = 0.0f;",
            f"if ({guard}{lane} < {P}) {{",
            f"    {coord} = {self.offset} + {lane};",
            f"    {value} = {self.Z}.data[({self.group} * {self.rows} + {row}) * {P} + {lane}];",
            f"}} else if ({guard}{lane} <= {P + 1}) {{",
            f"    const int sparse_entry = ({self.base} + {row}) * 2 + {lane} - {P};",
            f"    {coord} = {self.sparse_dof}.data[sparse_entry];",
            f"    {value} = {self.sparse_z}.data[sparse_entry];",
            "}",
        ]
        return "\n".join(indent + line for line in lines)

    def _slot(self, row: str, name: str, indent: str) -> str:
        """One thread: declare ``name``, the entry of row ``row``'s slot that holds its coordinate."""
        sparse_row = f"({self.base} + {row})"
        return f"{indent}const int {name} = {sparse_row} * 2 + ({self.sparse_dof}.data[{sparse_row} * 2] >= 0 ? 0 : 1);"

    def _here(self, sibling: str, out: str) -> list[str]:
        """A slot lane's entry of row ``sibling`` on the lane's own coordinate ``coord``: the sibling's slot that
        carries it, zero when none does (``out`` declared zero before)."""
        lane, P = self.lanes.lane, self.dofs
        return [
            f"if ({lane} >= {P} && {lane} <= {P + 1}) {{",
            f"    const int {out}_row = ({self.base} + {sibling}) * 2;",
            f"    {out} = 0.0f;",
            f"    if (coord >= 0 && coord == {self.sparse_dof}.data[{out}_row])",
            f"        {out} = {self.sparse_z}.data[{out}_row];",
            f"    else if (coord >= 0 && coord == {self.sparse_dof}.data[{out}_row + 1])",
            f"        {out} = {self.sparse_z}.data[{out}_row + 1];",
            "}",
        ]

    def consume(self, row: str, indent: str) -> str:
        """Declare the visited row ``row``'s entries: this lane's ``coord`` and ``z``."""
        if not self.lanes.parallel:
            return f"{self._slot(row, 'row_entry', indent)}\n{indent}const float z = {self.sparse_z}.data[row_entry];"
        return self._entries(row, "coord", "z", indent)

    def residual(self, out: str, indent: str) -> str:
        """``float out = Z_i^T u`` from the consumed entries (every lane holds it)."""
        if not self.lanes.parallel:
            return f"{indent}float {out} = __fmul_rn(z, {self.velocity.at(self.coordinate)});"
        return (
            f"{indent}float {out}_partial = coord >= 0 ? z * {self.velocity.at('coord')} : 0.0f;\n"
            f"{indent}{self.lanes.reduce(f'{out}_partial')}\n"
            f"{indent}float {out} = {out}_partial;"
        )

    def row_velocity(self, out: str, row: str, indent: str, *, active: str | None = None) -> str:
        """``float out = Z_row^T u`` of a row outside the visit (``active``: whether this lane's row exists)."""
        if not self.lanes.parallel:
            return (
                f"{self._slot(row, f'{out}_entry', indent)}\n"
                f"{indent}float {out} = __fmul_rn({self.sparse_z}.data[{out}_entry], "
                f"{self.velocity.at(self.coordinate)});"
            )
        return (
            self._entries(row, f"{out}_coord", f"{out}_z", indent, active)
            + f"\n{indent}float {out} = {out}_coord >= 0 ? {out}_z * {self.velocity.at(f'{out}_coord')} : 0.0f;"
            + f"\n{indent}{self.lanes.reduce(out)}"
        )

    def pair_entries(self, sibling: str, indent: str) -> str:
        """Load the sibling's entries once for the pair's terms and its apply: one thread's slot entry (``sib_z``); a
        lane's coordinate and entry of the sibling, and the sibling's entry on the lane's own coordinate
        (``sib_z_here``: a sparse slot's when the sibling carries that coordinate)."""
        if not self.lanes.parallel:
            return (
                f"{self._slot(sibling, 'sib_entry', indent)}\n"
                f"{indent}const float sib_z = {self.sparse_z}.data[sib_entry];"
            )
        lines = [
            self._entries(sibling, "sib_coord", "sib_z", ""),
            "float sib_z_here = sib_z;",
            *self._here(sibling, "sib_z_here"),
        ]
        return "\n".join(indent + line for line in "\n".join(lines).split("\n"))

    def pair_terms(self, row: str, sibling: str, rhs: Callable[[str], str], indent: str, *, own: bool = False) -> str:
        """The pair's terms from the sibling's entries (:meth:`pair_entries`): one thread's two products, a lane's
        two products reduced in one ladder."""
        if own:
            return super().pair_terms(row, sibling, rhs, indent, own=own)
        if not self.lanes.parallel:
            return (
                f"{indent}float sibling_residual = __fmul_rn(sib_z, {self.velocity.at(self.coordinate)});\n"
                f"{indent}sibling_residual += {rhs(sibling)};\n"
                f"{indent}float cross = z * sib_z;"
            )
        return (
            f"{indent}float sibling_residual = sib_coord >= 0 ? sib_z * {self.velocity.at('sib_coord')} : 0.0f;\n"
            f"{indent}float cross = coord >= 0 ? z * sib_z_here : 0.0f;\n"
            f"{indent}{self.lanes.reduce('sibling_residual', 'cross')}\n"
            f"{indent}sibling_residual += {rhs(sibling)};"
        )

    def spin_terms(self, row: str, sibling: str, spin: str, rhs: Callable[[str], str], indent: str) -> str:
        """The spin row's terms (contact torsion): its residual, and its couplings with the row and the sibling, each
        an entry product on the visited row's coordinate, which a contact's rows share (one thread: its slot's)."""
        if not self.lanes.parallel:
            return (
                f"{self._slot(spin, 'spin_entry', indent)}\n"
                f"{indent}const float spin_z = {self.sparse_z}.data[spin_entry];\n"
                f"{indent}float spin_residual = __fmul_rn(spin_z, {self.velocity.at(self.coordinate)});\n"
                f"{indent}spin_residual += {rhs(spin)};\n"
                f"{indent}float spin_cross = z * spin_z;\n"
                f"{indent}float spin_cross_sibling = sib_z * spin_z;"
            )
        lines = [
            self._entries(spin, "spin_coord", "spin_z", ""),
            "float spin_z_here = spin_z;",
            *self._here(spin, "spin_z_here"),
            f"float spin_residual = spin_coord >= 0 ? spin_z * {self.velocity.at('spin_coord')} : 0.0f;",
            "float spin_cross = coord >= 0 ? z * spin_z_here : 0.0f;",
            "float spin_cross_sibling = coord >= 0 ? sib_z_here * spin_z_here : 0.0f;",
            self.lanes.reduce("spin_residual", "spin_cross", "spin_cross_sibling"),
            f"spin_residual += {rhs(spin)};",
        ]
        return "\n".join(indent + line for line in "\n".join(lines).split("\n"))

    def coupling(self, out: str, row: str, sibling: str, indent: str) -> str:
        """``float out = Z_i^T Z_s`` of the visited row ``row``: each lane's entry times the sibling's on its
        coordinate (a sparse slot's, when the sibling carries that coordinate), reduced."""
        if not self.lanes.parallel:
            return (
                f"{self._slot(sibling, f'{out}_entry', indent)}\n"
                f"{indent}float {out} = z * {self.sparse_z}.data[{out}_entry];"
            )
        lane, P = self.lanes.lane, self.dofs
        lines = [
            f"float {out}_here = 0.0f;",
            f"if ({lane} < {P}) {out}_here = {self.Z}.data[({self.group} * {self.rows} + {sibling}) * {P} + {lane}];",
            *self._here(sibling, f"{out}_here"),
            f"float {out} = coord >= 0 ? z * {out}_here : 0.0f;",
            self.lanes.reduce(out),
        ]
        return "\n".join(indent + line for line in lines)

    def apply(self, delta: str, indent: str) -> str:
        """``u += Z_i delta`` from the consumed entries."""
        if not self.lanes.parallel:
            return f"\n{indent}{self.velocity.at(self.coordinate)} += z * {delta};"
        return f"\n{indent}if (coord >= 0) {self.velocity.at('coord')} += z * {delta};"

    def apply_row(self, row: str, delta: str, indent: str) -> str:
        """``u += Z_row delta`` for a row outside the visit; the pair's sibling ``sib`` from its kept entries
        (:meth:`pair_entries`)."""
        if row == "sib":
            if not self.lanes.parallel:
                return f"\n{indent}{self.velocity.at(self.coordinate)} += sib_z * {delta};"
            return f"\n{indent}if (sib_coord >= 0) {self.velocity.at('sib_coord')} += sib_z * {delta};"
        if not self.lanes.parallel:
            return (
                f"\n{self._slot(row, 'apply_entry', indent)}\n"
                f"{indent}{self.velocity.at(self.coordinate)} += {self.sparse_z}.data[apply_entry] * {delta};"
            )
        lines = [
            self._entries(row, "apply_coord", "apply_z", ""),
            f"if (apply_coord >= 0) {self.velocity.at('apply_coord')} += apply_z * {delta};",
        ]
        return "".join(f"\n{indent}{line}" for line in "\n".join(lines).split("\n"))


@dataclass(frozen=True)
class LocalBlock(ResidualBlock):
    """The response block of a local owner's rows, whitened by the walk and stored as a packed lower triangle (FL3.6).

    The owner's ``dense`` rows act on its ``dofs`` coordinates: its tree's ``primary`` coordinates, in the tree's
    coordinate order, then a paired free body's ``secondary`` six, a one-body tree of the same walk. The block holds
    each row's footprints ``z``, staged in ``s_Z`` (row ``r`` from ``r * dofs``, :meth:`respond`): the walk of the
    row's seeds in the kernel (:class:`.row_inputs.CompactLocalRows`), or the one writer's stored rows
    (:class:`.row_inputs.StoredLocalRows`). With ``bodies`` (at most the primary tree's bodies) the kernel walks
    seeds, and the walk's data of the world's trees is staged in shared memory once (``s_walk``, ``s_walk_path``:
    :meth:`walk_floats`, :meth:`walk_ints`; with ``wrenches`` the step's columns and links too, for wrench seeds);
    without, it is read in place. With ``release`` the class's drives can slip: a walked row takes the release ``R`` of
    its tree's slips along their paths (``WALK_RELEASE``, the one writer's rows carry it already; the step's one leave
    takes ``R^T``, :func:`.tree_factor.walk_leave_kernel`).
    With ``free``, the world's free-body rows (``free_count``, :class:`FreeBodyZ` entries read in place, their
    ``mf_meta`` words from ``off_meta``) follow the dense rows in the block: they act on the paired free body, whose
    coordinates are the last block's, so a free row's terms are its ``Z`` against the body's ``u`` and the dense rows'
    ``Z`` there.

    ``s_A`` holds the lower triangle of the Gram matrix ``A = Z^T Z``:
    ``A[hi][lo] = Z_hi Z_lo`` (``hi >= lo``) at ``hi (hi + 1) / 2 + lo``, built
    lane-strided over the packed entries, each a sum in coordinate order from
    zero (:meth:`assemble`); either row of an entry reads it. The residuals
    start at the rows' right-hand sides, their incident velocities included.
    The velocity change is committed once after the sweeps, ``u = sum_r Z_r
    applied_r`` over every row of the block in row order (:meth:`commit`), and goes to
    the step's one leave in the walk's coordinates (:attr:`.tree_factor.WalkTrees.du`). A view of the free-body rows
    addresses them from block row ``first``.
    """

    dofs: int
    primary: int = 0
    bodies: int = 0
    seeds: bool = False
    staged: bool = False
    wrenches: bool = False
    release: bool = False
    lanes: Lanes = Lanes()
    dense: str = "row_count"
    free: FreeBodyZ | None = None
    free_count: str = "mf_count"
    first: str = ""
    diagonal: Callable[[str], str] | None = None
    inverse: Callable[[str], str] | None = None
    name = "local_block"

    @property
    def rows(self) -> str:
        """The block's rows: the dense rows, then the free-body rows."""
        return self.dense if self.free is None else f"{self.dense} + {self.free_count}"

    def block_row(self, row: str) -> str:
        return f"{self.first} + {row}" if self.first else row

    def entry(self, source: str, target: str) -> tuple[str, str]:
        a, b = _operand(source), _operand(target)
        return f"const int hi = {a} > {b} ? {a} : {b}, lo = {a} > {b} ? {b} : {a};", "hi * (hi + 1) / 2 + lo"

    @property
    def secondary(self) -> int:
        return self.dofs - self.primary

    @property
    def coordinate_floats(self) -> int:
        """Staged floats a coordinate: ``s``, ``ut``, ``1 / ell`` (its record unpacked, :data:`.tree_scan.WALK_STAGED`),
        then with ``wrenches`` the step's column the finish published, unpacked (six floats, :data:`.tree_factor.WALK_NOW`)."""
        from .tree_scan import WALK_STAGED  # noqa: PLC0415 (tree_factor imports this module)

        return WALK_STAGED + (6 if self.wrenches else 0)

    @property
    def link_floats(self) -> int:
        """Staged floats a body: its offset to its parent's frame, then with ``wrenches`` its frame point (the factor's,
        where the step's columns are shifted)."""
        return 6 if self.wrenches else 3

    def walk_floats(self) -> int:
        """Floats a lane group stages of the walk: every coordinate's, then every primary body's (and with
        ``wrenches`` the free body's)."""
        free = 1 if self.wrenches and self.secondary else 0
        return self.dofs * self.coordinate_floats + (self.bodies + free) * self.link_floats if self.staged else 0

    def walk_ints(self) -> int:
        """Ints a lane group stages: the primary's path table, the free body's entry, each primary coordinate's body."""
        return 2 * self.bodies + 2 + self.primary if self.staged else 0

    def walk_preludes(self) -> tuple:
        """The walk's functions over this block's layout (:func:`.tree_factor.walk_paths_prelude`: the staged copy's
        strides, or the device walk's), and with ``seeds`` the seed walk its rows' walks bind
        (:func:`.tree_factor.walk_row_prelude`)."""
        from .tree_factor import walk_paths_prelude, walk_row_prelude  # noqa: PLC0415 (tree_factor imports this module)

        walk = walk_paths_prelude(1)
        if self.staged:
            walk = walk_paths_prelude(
                1, coordinate=self.coordinate_floats, link=self.link_floats, now=self.coordinate_floats
            )
        return (walk, walk_row_prelude(1)) if self.seeds else (walk,)

    def prologue(self) -> str:
        """Bind the world's trees on the walk (``walk``, :class:`.tree_factor.WalkTrees`; articulation ``art``, and
        ``secondary_art`` with a free body): their data, links and path tables, staged here once (``staged``, the
        primary tree of at most ``bodies`` bodies) or in place; ``walk_primary()`` and ``walk_free()`` select a tree for
        the walk's functions; and with ``seeds`` the row seeds' walks a lane runs into ``s_Z`` (``walk_covector`` and,
        with ``wrenches``, ``walk_wrench``: each returns its incident velocity, the step's ``J`` entries against the
        staged velocity)."""
        from .tree_factor import (  # noqa: PLC0415 (tree_factor imports this module)
            WALK_COORDINATE,
            WALK_LINK,
            WALK_NOW,
            WALK_UNPACKED,
        )
        from .tree_scan import WALK_STAGED  # noqa: PLC0415

        P, S, L = self.primary, self.secondary, self.lanes.width
        lane = self.lanes.lane
        lines = [
            "const int walk_j0 = walk.joint_start.data[art], walk_d0 = walk.dof_start.data[art];",
            "const int walk_bodies = walk.joint_end.data[art] - walk_j0;",
        ]
        if S:
            lines.append(
                "const int free_j0 = walk.joint_start.data[secondary_art], free_d0 = walk.dof_start.data[secondary_art];"
            )
        if not self.staged:
            lines += [
                f"const float* walk_data = walk.data.data + (size_t)walk_d0 * {WALK_COORDINATE};",
                f"const float* walk_link = walk.link.data + (size_t)walk_j0 * {WALK_LINK};",
                "const int* walk_path = reinterpret_cast<const int*>(walk.path.data + walk_j0);",
            ]
            if self.seeds:
                lines.append("const int* walk_dof_body = walk.dof_body.data + walk_d0;")
            if self.wrenches and self.seeds:
                lines += [
                    f"const float* walk_now = walk.now.data + (size_t)walk_d0 * {WALK_NOW};",
                    f"const float* walk_now_link = walk.now_link.data + (size_t)walk_j0 * {WALK_LINK};",
                    "float walk_shift[3];",
                    "WALK_SHIFT(walk, art, walk_shift);",
                ]
            if S:
                lines += [
                    f"const float* free_data = walk.data.data + (size_t)free_d0 * {WALK_COORDINATE};",
                    f"const float* free_link = walk.link.data + (size_t)free_j0 * {WALK_LINK};",
                    "const int* free_path = reinterpret_cast<const int*>(walk.path.data + free_j0);",
                ]
                if self.wrenches and self.seeds:
                    lines += [
                        f"const float* free_now = walk.now.data + (size_t)free_d0 * {WALK_NOW};",
                        f"const float* free_now_link = walk.now_link.data + (size_t)free_j0 * {WALK_LINK};",
                        "float free_shift[3];",
                        "WALK_SHIFT(walk, secondary_art, free_shift);",
                    ]
        else:
            C, K, B = self.coordinate_floats, self.link_floats, self.bodies
            coordinate = "walk_d0 + c" if not S else f"c < {P} ? walk_d0 + c : free_d0 + c - {P}"
            now = (
                [
                    "    float S[6];",
                    f"    WALK_LOAD_COLUMN(S, walk.now.data + (size_t)d * {WALK_NOW});",
                    f"    for (int k = 0; k < 6; ++k) s_walk[c * {C} + {WALK_STAGED} + k] = S[k];",
                ]
                if self.wrenches
                else []
            )
            lines += [
                "// The walk's data of the world's trees staged once (FL3_design section 4), every load in flight together:",
                "// each coordinate's s, ut and 1 / ell unpacked (and the step's column the finish published), each body's link offset",
                "// (and its frame point), the path tables and each primary coordinate's body.",
                f"for (int c = {lane}; c < {self.dofs}; c += {L}) {{",
                f"    const int d = {coordinate};",
                f"    float r[{WALK_UNPACKED}];",
                f"    WALK_LOAD_RECORD(r, walk.data.data + (size_t)d * {WALK_COORDINATE});",
                f"    for (int k = 0; k < {WALK_STAGED}; ++k) s_walk[c * {C} + k] = r[k];",
                *now,
                "}",
                f"for (int b = {lane}; b < walk_bodies; b += {L}) {{",
                "    float l[6];",
                f"    WALK_LOAD_LINK(l, walk.link.data + (size_t)walk_j0 * {WALK_LINK}, b,",
                "                   reinterpret_cast<const int*>(walk.path.data + walk_j0)[2 * b]);",
                f"    for (int k = 0; k < {K}; ++k) s_walk[{self.dofs * C} + b * {K} + k] = l[k];",
                "}",
                f"for (int i = {lane}; i < 2 * walk_bodies; i += {L})",
                "    s_walk_path[i] = reinterpret_cast<const int*>(walk.path.data + walk_j0)[i];",
                f"for (int i = {lane}; i < {P}; i += {L}) s_walk_path[{2 * B + 2} + i] = walk.dof_body.data[walk_d0 + i];",
                f"if ({lane} == 0) {{",
                f"    s_walk_path[{2 * B}] = -1;",
                f"    s_walk_path[{2 * B + 1}] = 0;",
                "}",
                "const float* walk_data = s_walk;",
                f"const float* walk_link = s_walk + {self.dofs * C};",
                "const int* walk_path = s_walk_path;",
                f"const int* walk_dof_body = s_walk_path + {2 * B + 2};",
            ]
            if self.wrenches:
                lines += [
                    f"const float* walk_now = s_walk + {WALK_STAGED};",
                    f"const float* walk_now_link = walk.now_link.data + (size_t)walk_j0 * {WALK_LINK};",
                    "float walk_shift[3];",
                    "WALK_SHIFT(walk, art, walk_shift);",
                ]
            if S:
                lines += [
                    f"const float* free_data = s_walk + {P * C};",
                    "const float* free_link = walk_link;",
                    f"const int* free_path = s_walk_path + {2 * B};",
                ]
                if self.wrenches:
                    # The free body's own link: its frame point, where its step columns are shifted.
                    lines += [
                        f"if ({lane} == 0) {{",
                        "    float l[6];",
                        f"    WALK_LOAD_LINK(l, walk.link.data + (size_t)free_j0 * {WALK_LINK}, 0, -1);",
                        f"    for (int k = 0; k < {K}; ++k) s_walk[{self.dofs * C + B * K} + k] = l[k];",
                        "}",
                        f"const float* free_now = free_data + {WALK_STAGED};",
                        f"free_link = s_walk + {self.dofs * C + B * K};",
                        f"const float* free_now_link = walk.now_link.data + (size_t)free_j0 * {WALK_LINK};",
                        "float free_shift[3];",
                        "WALK_SHIFT(walk, secondary_art, free_shift);",
                    ]
        lines.append(f"const auto walk_primary = [&]() {{ WALK_TREE(walk_path, walk_bodies, {P}); }};")
        if S:
            lines.append(f"const auto walk_free = [&]() {{ WALK_TREE(free_path, 1, {S}); }};")
        if self.seeds:
            lines += [
                "// A covector seed weight e_dof of the primary tree walked into the row's footprints z; its incident.",
                "const auto walk_covector = [&](float* z, const int dof, const float weight) -> float {",
                "    float incident = 0.0f;",
                "    const float x[1] = {weight};",
                "    WALK_ROW(walk_data, walk_link, nullptr, nullptr, nullptr, walk_dof_body[dof], dof, x,",
                "             [&](int, int c, float y, float j) {",
                "        z[c] += y;",
                "        incident += j * s_v[c];",
                "    });",
                "    return incident;",
                "};",
            ]
            if self.wrenches:
                lines += [
                    "// A wrench seed at body b's walk frame (the factor's), walked into the footprints z of a tree (data,",
                    "// link, now, now_link, shift: its walk, the step's geometry the finish published and its origins'",
                    "// difference; v its staged velocity); its incident.",
                    "const auto walk_wrench = [&](float* z, const float* v, const float* data, const float* link,",
                    "                             const float* now, const float* now_link, const float* shift, const int b,",
                    "                             const float* wrench) -> float {",
                    "    float incident = 0.0f;",
                    "    WALK_ROW(data, link, now, now_link, shift, b, -1, wrench, [&](int, int c, float y, float j) {",
                    "        z[c] += y;",
                    "        incident += j * v[c];",
                    "    });",
                    "    return incident;",
                    "};",
                ]
        return "\n".join(lines)

    def respond(self, source, indent: str) -> str:
        """Stage row ``row``'s footprints ``z`` in ``s_Z``, in the caller's loop over ``row``: the row source's
        (``source.stage``: the walk of its seeds, or the one writer's stored rows) and declare ``active`` (some entry is
        nonzero), ``incident`` (its ``J v_hat``) and ``row_diagonal`` (``|z|^2``, in coordinate order)."""
        lines = [
            f"const int staged = row * {self.dofs};",
            source.stage(self),
            "int active = 0;",
            "float row_diagonal = 0.0f;",
            f"for (int c = 0; c < {self.dofs}; ++c) {{",
            "    const float z = s_Z[staged + c];",
            "    row_diagonal += z * z;",
            "    if (z != 0.0f) active = 1;",
            "}",
        ]
        return "\n".join(indent + line if line.strip() else line for line in "\n".join(lines).split("\n"))

    def _free_row(self, free: str, name: str, indent: str, *, word: bool = False) -> str:
        """Decode free-body row ``free``'s bodies (``{name}_a``, ``{name}_b``) and entry base (``{name}6``).

        With ``word`` the row's whole packed word is read (``{name}_meta``: bodies, inverse diagonal, rhs, type).
        """
        packed = f"mf_meta.data[off_meta + ({free}) * 4]"
        read = (
            f"{indent}const int4 {name}_meta = *reinterpret_cast<const int4*>(&{packed});\n"
            f"{indent}const int {name}_dofs = {name}_meta.x;\n"
            if word
            else f"{indent}const int {name}_dofs = {packed};\n"
        )
        return (
            f"{read}"
            f"{indent}const int {name}_a = {name}_dofs >> 16;\n"
            f"{indent}const int {name}_b = ({name}_dofs << 16) >> 16;\n"
            f"{indent}const int {name}6 = {self.free.base} + ({free}) * 6;"
        )

    def _free_terms(self, out: str, other: Callable[[str, str], str], indent: str) -> str:
        """Add free-body row ``row``'s products with a vector in the block's factor coordinates to ``out``: each side's
        ``Z`` against the vector on that side's body (``other(side, d)``: coordinate ``d`` of the body at ``side``),
        body ``a``'s side, then ``b``'s, each from its own sum."""
        free = self.free
        lines = []
        for Z, side in ((free.Z_a, "row_a"), (free.Z_b, "row_b")):
            lines += [
                f"if ({side} >= 0) {{",
                "    float side_term = 0.0f;",
                f"    for (int d = 0; d < 6; ++d) side_term += {Z}.data[row6 + d] * {other(side, 'd')};",
                f"    {out} += side_term;",
                "}",
            ]
        return "\n".join(indent + line for line in "\n".join(lines).split("\n"))

    def assemble(self, rhs: Vector, indent: str) -> str:
        """Start every row's residual at its right-hand side (``s_res``, ``s_app`` cleared) and build ``A = Z^T Z``'s
        lower triangle.

        A dense row's products run over the block's coordinates, a free-body row's over its bodies' six
        (:meth:`_free_terms`), body ``a``'s side and then ``b``'s; a free-body row's response to a free-body column sums
        the column's bodies acting on the row's body, ``Z`` against ``Z``.
        """
        lane, width, T, free = self.lanes.lane, self.lanes.width, self.dofs, self.free
        p = indent
        dense_residual = _at(rhs, "r")
        lines = [f"{p}{{", f"{p}    const int block_rows = {self.rows};"]
        lines.append(f"{p}    for (int r = {lane}; r < block_rows; r += {width}) {{")
        if free is None:
            lines.append(f"{p}        const float residual = {dense_residual};")
        else:
            free_word = f"mf_meta.data[off_meta + (r - {self.dense}) * 4 + 2]"
            lines.append(
                f"{p}        const float residual = r < {self.dense} ? {dense_residual} : __int_as_float({free_word});"
            )
        lines += [
            f"{p}        s_res[r] = residual;",
            f"{p}        s_app[r] = 0.0f;",
            f"{p}    }}",
            f"{p}    const int entries = block_rows * (block_rows + 1) / 2;",
            f"{p}    int hi = 0, start = 0, next = 1;",
            f"{p}    for (int e = {lane}; e < entries; e += {width}) {{",
            f"{p}        while (e >= next) {{",
            f"{p}            start = next;",
            f"{p}            ++hi;",
            f"{p}            next += hi + 1;",
            f"{p}        }}",
            f"{p}        const int lo = e - start;",
            f"{p}        float a = 0.0f;",
        ]
        dense_entry = f"for (int d = 0; d < {T}; ++d) a += s_Z[hi * {T} + d] * s_Z[lo * {T} + d];"
        if free is None:
            lines.append(f"{p}        {dense_entry}")
        else:
            q = p + "                "
            lines += [
                f"{p}        if (hi < {self.dense}) {{",
                f"{p}            {dense_entry}",
                f"{p}        }} else {{",
                self._free_row(f"hi - {self.dense}", "row", p + "            ", word=True),
                f"{p}            if (lo < {self.dense}) {{",
                self._free_terms("a", lambda side, d: f"s_Z[lo * {T} + {side} + {d}]", q),
                f"{p}            }} else {{",
                self._free_row(f"lo - {self.dense}", "column", q, word=True),
                f"{q}for (int d = 0; d < 6; ++d) {{",
                f"{q}    if (row_a >= 0) {{",
                f"{q}        float response = 0.0f;",
                f"{q}        if (column_a == row_a) response += {free.Z_a}.data[column6 + d];",
                f"{q}        if (column_b == row_a) response += {free.Z_b}.data[column6 + d];",
                f"{q}        a += {free.Z_a}.data[row6 + d] * response;",
                f"{q}    }}",
                f"{q}    if (row_b >= 0) {{",
                f"{q}        float response = 0.0f;",
                f"{q}        if (column_a == row_b) response += {free.Z_a}.data[column6 + d];",
                f"{q}        if (column_b == row_b) response += {free.Z_b}.data[column6 + d];",
                f"{q}        a += {free.Z_b}.data[row6 + d] * response;",
                f"{q}    }}",
                f"{q}}}",
                f"{p}            }}",
                f"{p}        }}",
            ]
        lines += [f"{p}        s_A[e] = a;", f"{p}    }}", f"{p}}}"]
        return "\n".join(lines)

    def commit(self, indent: str) -> str:
        """``u = sum_r Z_r applied_r`` over every row of the block, in row order (zero terms included), the free rows'
        on their body after the dense rows'; the caller then leaves factor coordinates (:meth:`leave`)."""
        lane, width, T, free = self.lanes.lane, self.lanes.width, self.dofs, self.free
        p = indent
        lines = [
            f"{p}for (int d = {lane}; d < {T}; d += {width}) {{",
            f"{p}    float u = 0.0f;",
            f"{p}    for (int r = 0; r < {self.dense}; ++r) u += s_Z[r * {T} + d] * s_app[r];",
        ]
        if free is not None:
            lines += [
                f"{p}    for (int f = 0; f < {self.free_count}; ++f) {{",
                self._free_row("f", "row", p + "        "),
                f"{p}        const float applied = s_app[{self.dense} + f];",
                f"{p}        if (row_a >= 0 && d >= row_a && d < row_a + 6) u += {free.Z_a}.data[row6 + d - row_a] * applied;",
                f"{p}        if (row_b >= 0 && d >= row_b && d < row_b + 6) u += {free.Z_b}.data[row6 + d - row_b] * applied;",
                f"{p}    }}",
            ]
        lines += [f"{p}    s_v[d] = u;", f"{p}}}"]
        return "\n".join(lines)


@dataclass(frozen=True)
class FactorComponent:
    """One diagonal block of a block-diagonal factor: ``size`` coordinates from lane ``offset``.

    ``lane`` is this lane's coordinate in the block (``LANE - offset``, out of range on other lanes). A lane's coordinate
    is a physical position: factor coordinate ``i`` sits on lane ``size - 1 - i`` (:func:`leaf_first`).
    """

    size: int
    offset: str
    lane: str


@dataclass(frozen=True)
class FactorRows(ResponseOperator):
    """Rows in factor coordinates: ``Z = L^-1 J^T`` over ``dofs`` coordinates (at most a warp), one lane each.

    With ``H = L L^T`` and the velocity change ``u = L^T dv``, a row's velocity
    term is ``J_i dv = Z_i u`` and its response ``u += Z_i delta``: one row array
    (``rows``, row ``i`` from ``base + i * dofs``) is both. Lane ``l`` holds
    coordinate ``l`` of ``u`` in the register ``velocity``, and of the rows it
    acts on in registers: the current row's (``current``), its friction
    sibling's (``sibling``) and, under contact torsion, its contact's spin
    row's (``spin``), which the schedule loads (:meth:`load`, or the pipeline
    :meth:`prefetch` and :meth:`consume`). A sum is one product per
    lane (zero past ``dofs``) reduced with ``SUM``; a pair's sums reduce in one
    ladder (:meth:`pair_terms`). The builder stores ``A_ii``; the step divides
    by it. ``components``, the diagonal blocks of ``L``, map the velocity
    change back out of factor coordinates (:meth:`leave`).
    """

    dofs: int
    components: tuple[FactorComponent, ...] = ()
    lanes: Lanes = Lanes()
    rows: str = "factor_rows"
    base: str = "row_base"
    velocity: str = "factor_velocity"
    current: str = "row_z"
    sibling: str = "sibling_z"
    spin: str = "spin_z"
    diagonal: Callable[[str], str] | None = None
    inverse: Callable[[str], str] | None = None
    name = "factor_rows"

    def load(self, row: str) -> str:
        """This lane's coordinate of row ``row`` (zero past ``dofs``)."""
        lane = self.lanes.lane
        return f"{lane} < {self.dofs} ? {self.rows}.data[{self.base} + ({row}) * {self.dofs} + {lane}] : 0.0f"

    def prefetch(self, first: str, indent: str) -> str:
        """Declare the pipeline register ``pre_z`` holding row ``first``."""
        return f"{indent}float pre_z = {self.load(first)};"

    def consume(self, row: str, end: str, indent: str) -> str:
        """Take row ``row`` from the pipeline (``current``) and load row ``row + 1``'s (before ``end``)."""
        lane = self.lanes.lane
        return (
            f"{indent}const float {self.current} = pre_z;\n"
            f"{indent}if ({row} + 1 < {end} && {lane} < {self.dofs})\n"
            f"{indent}    pre_z = {self.rows}.data[{self.base} + ({row} + 1) * {self.dofs} + {lane}];"
        )

    def _product(self, out: str, a: str, b: str) -> str:
        return f"float {out} = {self.lanes.lane} < {self.dofs} ? {a} * {b} : 0.0f;"

    def _sums(self, channels, indent: str, *, magnitudes: tuple[str, ...] = ()) -> str:
        """Reduce each channel ``(out, a, b)``'s products in one ladder; a velocity sum (``b`` the velocity), and each
        channel named in ``magnitudes``, reduces its terms' magnitude beside it, ``{out}_terms`` (:meth:`terms`)."""
        lines, names = [], []
        for out, a, b in channels:
            lines.append(f"{indent}{self._product(out, a, b)}")
            names.append(out)
            if b == self.velocity or out in magnitudes:
                lines.append(f"{indent}float {out}_terms = fabsf({out});")
                names.append(f"{out}_terms")
        return "\n".join([*lines, f"{indent}{self.lanes.reduce(*names)}"])

    def terms(self, out: str, rhs: str, *, whole: bool) -> str:
        """``sum_k |Z_ik u_k|`` of a velocity sum reduced into ``out``, reduced beside it."""
        return f"{out}_terms"

    def residual(self, out: str, indent: str) -> str:
        """``float out = Z_i u`` of the current row, and ``{out}_terms``."""
        return self._sums(((out, self.current, self.velocity),), indent)

    def sibling_residual(self, out: str, sibling: str, indent: str, rhs: str) -> str:
        """``float out = Z_s u + rhs`` of the friction sibling (its entries in the ``sibling`` register)."""
        return f"{self._sums(((out, self.sibling, self.velocity),), indent)}\n{indent}{out} += {rhs};"

    def coupling(self, out: str, row: str, sibling: str, indent: str) -> str:
        """``float out = Z_i Z_s``, the pair's off-diagonal entry."""
        return self._sums(((out, self.current, self.sibling),), indent)

    def pair_terms(self, row: str, sibling: str, rhs: Callable[[str], str], indent: str, *, own: bool = False) -> str:
        """The pair's sums in one ladder: the row's residual (with ``own``), the sibling's, then the coupling."""
        channels = [("residual", self.current, self.velocity)] if own else []
        channels += [("sibling_residual", self.sibling, self.velocity), ("cross", self.current, self.sibling)]
        lines = [self._sums(channels, indent)]
        if own:
            lines.append(f"{indent}residual += {rhs(row)};")
        lines.append(f"{indent}sibling_residual += {rhs(sibling)};")
        return "\n".join(lines)

    def spin_terms(self, row: str, sibling: str, spin: str, rhs: Callable[[str], str], indent: str) -> str:
        """The spin row's sums in one ladder (its entries in the ``spin`` register): its residual, then its couplings
        with the current row and the sibling."""
        channels = (
            ("spin_residual", self.spin, self.velocity),
            ("spin_cross", self.current, self.spin),
            ("spin_cross_sibling", self.sibling, self.spin),
        )
        return f"{self._sums(channels, indent)}\n{indent}spin_residual += {rhs(spin)};"

    def apply(self, delta: str, indent: str) -> str:
        """``u += Z_i delta`` of the current row (lanes past ``dofs`` add zero)."""
        return f"\n{indent}{self.velocity} += {self.current} * {delta};"

    def apply_row(self, row: str, delta: str, indent: str) -> str:
        """``u += Z_s delta`` of the friction sibling ``row`` (its entries in the ``sibling`` register), or of the spin
        row (``spin``, in the ``spin`` register)."""
        register = self.spin if row == "spin" else self.sibling
        return f"\n{indent}{self.velocity} += {register} * {delta};"

    def lanes_of(self, indent: str) -> list[str]:
        """Lines declaring each component's lane (``LANE - offset``)."""
        return [f"{indent}const int {c.lane} = {self.lanes.lane} - {c.offset};" for c in self.components]


@dataclass(frozen=True)
class SparseFactor(ResponseOperator):
    """Rows of ``Z = L^-1 P J^T`` as packed factor rows, acting on the factor-coordinate velocity delta.

    Row ``r`` packs ``support`` entries: entry ``e`` is ``factor(r, e)`` at the factor coordinate ``index(r, e)``,
    negative for padding, and the row's velocity term is ``Z_r . delta`` over the staged delta (``velocity``). With
    ``dense`` coordinates the packed rows are expanded once per call (:meth:`expand`) into rows of every coordinate
    in private shared storage (``local``): entry ``e`` at coordinate ``e``.

    With lanes, lane ``l`` owns entries ``l + q L`` (``q`` ascending): it keeps their coordinates and values in
    registers (``indices[q]``, ``responses[q]``) for the row's apply, its partial sums accumulate them from zero, and
    the lanes reduce with ``SUM``. One thread (``Lanes(1, "0")``) sums a row's entries in order. A friction pair's
    second tangent shares its first tangent's support: the first tangent's residual reduces the second's velocity
    term (``sibling_jv``) and the pair's coupling ``Z_t . Z_s`` (``pair_coupling``) in the same ``SUM``, and the
    apply adds the second tangent's change, then the first's, to each entry, written back once. The builder stores
    ``A_ii = |Z_i|^2 + cfm`` as a value: the step divides.
    """

    support: int
    index: Callable[[str, str], str]
    factor: Callable[[str, str], str]
    lanes: Lanes = Lanes()
    dense: int = 0
    local: str = "s_local_factor"
    velocity: Velocity = Velocity("s_delta")
    diagonal: Callable[[str], str] | None = None
    inverse: Callable[[str], str] | None = None
    name = "sparse_factor"
    fused_pair_apply = True

    @property
    def width(self) -> int:
        """Entries of a row as the sweeps read it: the packed support, or the dense coordinates."""
        return self.dense or self.support

    @property
    def entries(self) -> int:
        """Entries per lane."""
        return (self.width + self.lanes.width - 1) // self.lanes.width

    def coordinate(self, row: str, entry: str = "entry") -> str:
        """The factor coordinate of row ``row``'s entry ``entry`` (negative for padding)."""
        return entry if self.dense else self.index(_operand(row), entry)

    def element(self, row: str, entry: str = "entry") -> str:
        """Row ``row``'s value at entry ``entry``."""
        if self.dense:
            return f"{self.local}[{_operand(row)} * {self.dense} + {entry}]"
        return self.factor(_operand(row), entry)

    def entries_loop(self, body: str, indent: str, *, entry: bool = True) -> str:
        """``body`` for each entry this lane owns, ``q`` ascending; ``entry`` declares the row entry."""
        first = f"{self.lanes.lane} + q * {self.lanes.width}" if self.lanes.parallel else "q"
        head = ["#pragma unroll", f"for (int q = 0; q < {self.entries}; ++q) {{"]
        if entry:
            head.append(f"    const int entry = {first};")
        lines = [*head, *("    " + line for line in body.split("\n")), "}"]
        return "\n".join(indent + line if line.strip() else line for line in lines)

    def guarded_coordinate(self, row: str) -> str:
        """The coordinate of this lane's ``entry`` of row ``row``, -1 past the row (with lanes)."""
        if not self.lanes.parallel:
            return self.coordinate(row)
        return f"entry < {self.width} ? {self.coordinate(row)} : -1"

    # -- row actions ---------------------------------------------------------------------------------------------------

    def residual(
        self,
        out: str,
        indent: str,
        *,
        row: str = "i",
        sibling: str | None = None,
        paired: str = "false",
        spin: str | None = None,
    ) -> str:
        """``float out = Z_row . delta`` (every lane holds it), the row's entries kept for :meth:`apply`.

        With ``sibling``, a ``paired`` row (a first tangent) also reduces ``sibling_jv = Z_sibling . delta`` and
        ``pair_coupling = Z_row . Z_sibling`` in the same ``SUM`` and keeps the sibling's entries for
        :meth:`apply_pair`; with its contact's ``spin`` row (contact torsion, on the same support) also ``spin_jv``,
        ``spin_coupling = Z_row . Z_spin`` and ``spin_coupling_sibling = Z_sibling . Z_spin`` (:meth:`spin_terms`).
        """
        count = self.entries
        lines = [f"int indices[{count}];", f"float responses[{count}];"]
        body = [
            f"const int index = {self.guarded_coordinate(row)};",
            f"const float value = index >= 0 ? {self.element(row)} : 0.0f;",
        ]
        if sibling is None:
            lines.append(f"float {out} = 0.0f;")
            body += [
                "indices[q] = index;",
                "responses[q] = value;",
                f"if (index >= 0) {out} += value * {self.velocity.at('index')};",
            ]
            reduce = self.lanes.reduce(out)
        else:
            lines += [f"float sibling_responses[{count}];", f"float {out} = 0.0f;"]
            lines += ["float sibling_jv = 0.0f;", "float pair_coupling = 0.0f;"]
            if spin is not None:
                lines.append("float spin_jv = 0.0f, spin_coupling = 0.0f, spin_coupling_sibling = 0.0f;")
            body += [
                f"const float sibling_value = {paired} && index >= 0",
                f"    ? {self.element(sibling)} : 0.0f;",
                *(
                    (f"const float spin_value = {paired} && index >= 0", f"    ? {self.element(spin)} : 0.0f;")
                    if spin is not None
                    else ()
                ),
                "indices[q] = index;",
                "responses[q] = value;",
                "sibling_responses[q] = sibling_value;",
                "if (index >= 0) {",
                f"    const float velocity = {self.velocity.at('index')};",
                f"    {out} += value * velocity;",
                "    sibling_jv += sibling_value * velocity;",
                "    pair_coupling += value * sibling_value;",
                *(
                    (
                        "    spin_jv += spin_value * velocity;",
                        "    spin_coupling += value * spin_value;",
                        "    spin_coupling_sibling += sibling_value * spin_value;",
                    )
                    if spin is not None
                    else ()
                ),
                "}",
            ]
            sums = ("spin_jv", "spin_coupling", "spin_coupling_sibling") if spin is not None else ()
            reduce = self.lanes.reduce(out, "sibling_jv", "pair_coupling", *sums)
        text = "\n".join(indent + line for line in lines) + "\n" + self.entries_loop("\n".join(body), indent)
        return text + "".join(f"\n{indent}{line}" for line in reduce.split("\n") if line)

    def sibling_residual(self, out: str, sibling: str, indent: str, rhs: str) -> str:
        """``float out = Z_s . delta + rhs`` of the pair's second tangent, whose sum :meth:`residual` reduced."""
        return f"{indent}float {out} = sibling_jv;\n{indent}{out} += {rhs};"

    def coupling(self, out: str, row: str, sibling: str, indent: str) -> str:
        """``float out = Z_t . Z_s``: the pair's off-diagonal entry, which :meth:`residual` reduced."""
        return f"{indent}float {out} = pair_coupling;"

    def spin_terms(self, row: str, sibling: str, spin: str, rhs: Callable[[str], str], indent: str) -> str:
        """The spin row's residual and couplings, which :meth:`residual` reduced with the pair's."""
        return (
            f"{indent}float spin_residual = spin_jv;\n{indent}spin_residual += {rhs(spin)};\n"
            f"{indent}float spin_cross = spin_coupling;\n{indent}float spin_cross_sibling = spin_coupling_sibling;"
        )

    def apply(self, delta: str, indent: str) -> str:
        """``delta_v += Z_row delta`` from the entries :meth:`residual` (or :meth:`row_velocity`) kept."""
        target = self.velocity.at("indices[q]")
        return self.entries_loop(f"if (indices[q] >= 0) {target} += responses[q] * {delta};", indent, entry=False)

    def apply_pair(self, sibling_delta: str, delta: str, indent: str) -> str:
        """``delta_v += Z_s sibling_delta + Z_row delta`` from :meth:`residual`'s entries: per entry the sibling's
        change, then the row's, each when nonzero, written back once."""
        target = self.velocity.at("indices[q]")
        body = (
            "if (indices[q] >= 0) {\n"
            f"    float value = {target};\n"
            f"    if ({sibling_delta} != 0.0f) value += sibling_responses[q] * {sibling_delta};\n"
            f"    if ({delta} != 0.0f) value += responses[q] * {delta};\n"
            f"    {target} = value;\n"
            "}"
        )
        return self.entries_loop(body, indent, entry=False)

    def accumulate(self, out: str, row: str, indent: str) -> str:
        """``out += Z_row . delta`` over this lane's entries, in order, unreduced."""
        body = (
            f"const int index = {self.guarded_coordinate(row)};\n"
            f"if (index >= 0) {out} += {self.element(row)} * {self.velocity.at('index')};"
        )
        return self.entries_loop(body, indent)

    def row_velocity(self, out: str, row: str, indent: str) -> str:
        """``float out = Z_row . delta`` (every lane holds it), the row's entries kept for :meth:`apply_strided`."""
        count = self.entries
        body = (
            f"const int index = {self.guarded_coordinate(row)};\n"
            f"const float response = index >= 0 ? {self.element(row)} : 0.0f;\n"
            "indices[q] = index;\n"
            "responses[q] = response;\n"
            f"if (index >= 0) {out} += response * {self.velocity.at('index')};"
        )
        reduce = self.lanes.reduce(out)
        return (
            f"{indent}int indices[{count}];\n{indent}float responses[{count}];\n{indent}float {out} = 0.0f;\n"
            + self.entries_loop(body, indent)
            + (f"\n{indent}{reduce}" if reduce else "")
        )

    def apply_strided(self, row: str, delta: str, indent: str) -> str:
        """``delta_v += Z_row delta`` from the entries :meth:`row_velocity` kept."""
        return self.apply(delta, indent)

    def apply_row(self, row: str, delta: str, indent: str) -> str:
        """``delta_v += Z_row delta``, the row's entries read here."""
        body = (
            f"const int index = {self.guarded_coordinate(row)};\n"
            "if (index >= 0) {\n"
            f"    const float response = {self.element(row)};\n"
            f"    {self.velocity.at('index')} += response * {delta};\n"
            "}"
        )
        return self.entries_loop(body, indent)

    def expand(self, rows: str, indent: str) -> str:
        """Expand the packed rows ``[0, rows)`` into the dense rows (``local``), zero elsewhere; ends unsynchronized."""
        lane, width, dense, support = self.lanes.lane, self.lanes.width, self.dense, self.support
        return (
            f"{indent}for (int i = {lane}; i < {rows} * {dense}; i += {width}) {self.local}[i] = 0.0f;\n"
            f"{indent}{self.lanes.sync()}\n"
            f"{indent}for (int i = {lane}; i < {rows} * {support}; i += {width}) {{\n"
            f"{indent}    const int row = i / {support}, entry = i % {support};\n"
            f"{indent}    const int index = {self.index('row', 'entry')};\n"
            f"{indent}    if (index >= 0) {self.local}[row * {dense} + index] = {self.factor('row', 'entry')};\n"
            f"{indent}}}"
        )
